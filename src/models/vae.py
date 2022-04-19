# -*- coding: utf-8 -*-
"""
module model.py
--------------------
Definition of the machine learning model for the task.
"""
import os
import sys
import tensorflow as tf
import numpy as np
import flow
from flow.config import Config
from flow.dataset import Dataset
from flow.callbacks import on_train_begin, on_epoch_begin
from flow.model import Inputs, Outputs
from .learner import DefaultLearner
from layers import EncoderBlock, DecoderBlock
# from metrics import f1 as metrics


class Vae(flow.Model):
    """
    Naive Unet model class. 
    Unet model is built based on the [the original paper](https://arxiv.org/pdf/1505.04597.pdf).
    """

    def __init__(self, inputs_config, config=None):
        """
        Model initialization function.
        :param inputs_config: a dictionary defining the shape and type of the model's inputs.
        """
        self.config = Config()
        self.init_configs()

        self.is_distributed = False
        if not self.is_distributed:
            self.batch_size = int(self.config.get("flow.batch_size"))
            self.global_batch_size = self.batch_size
            self.config["flow.global_batch_size"] = self.global_batch_size
            self.init_layers()
            self.global_step = tf.train.get_or_create_global_step()
            super().__init__(inputs_config, config)
            self._update_op = None
        else:
            self.inputs_config = inputs_config

            self.inputs = Inputs()
            # set in _ get_model
            self.outputs = Outputs()
            self._outs = dict()
            self._is_session_initialized = False
            # mirror distributed strategy
            self.distributed_strategy = tf.distribute.MirroredStrategy()
            self.batch_size = int(self.config.get("flow.batch_size"))
            self.global_batch_size = self.batch_size * self.distributed_strategy.num_replicas_in_sync
            self.config["flow.global_batch_size"] = self.global_batch_size

            with self.distributed_strategy.scope():
                self.init_layers()
                with tf.variable_scope("learning_phase", reuse=tf.AUTO_REUSE):
                    self._is_training_phase = tf.get_variable(
                        name="is_training_phase",
                        dtype=tf.bool,
                        initializer=True,
                        trainable=False,
                    )
                self.global_step = tf.train.get_or_create_global_step()

        self._callbacks()

    def init_configs(self):
        """
        gets the model's configs and saves into class parameters.
        """
        # task hyperparameters
        self.n_classes = int(self.config.get("flow.n_classes", 1))
        self.n_epochs = int(self.config.get("flow.n_epochs", 40))

        # optimizer
        self.lr_decay_type = self.config.get("optimizer.decay_type", None)
        self.initial_lr = float(self.config.get("optimizer.lr"))
        self.decay_period_in_epochs = int(self.config.get("optimizer.decay_period_in_epochs", 0))
        self.decay_rate = float(self.config.get("optimizer.decay_rate", 0.96))
        self.warmup_type = self.config.get("optimizer.warmup_type", None)
        self.warmup_steps_in_epochs = int(self.config.get("optmizer.warmup_steps_in_epochs", 0))
        self.gradient_clip = self.config.get("optimizer.gradient_clip", "false").lower().strip() == "true"
        self.clip_norm = float(self.config.get("optimizer.clip_norm", 40)) 

    def init_layers(self):
        # building reversible arquitecture.
        self._skips = list()
        self._encode_blocks = list()
        self._decode_blocks = list()
        
        self.encoder = EncoderBlock(
            filters=32, kernel_size=[3, 3], strides=[1, 1],
            dropout_rate=0., activation="relu",
            padding="SAME", kernel_initializer="he_normal", n_latent=7,
            name=None,
        )

        self.decoder = DecoderBlock(
            filters=32, kernel_size=[3, 3], strides=[1, 1],
            dropout_rate=0., activation="relu",
            padding="SAME", kernel_initializer="he_normal", n_latent=7,
            name=None
        )

    def get_model_spec(self, *args, **kwargs):
        """
        Builds the model tensors and layers.
        It also sets the appropriate outputs variables.
        :return: a compiled model
        """

        def step(*args):
            x, = args
            # add input tensors to collections, 
            # making them globally accessable
            tf.add_to_collection("flow_inputs", x)
            # tf.add_to_collection("flow_inputs", y)

            z, mean, std = self.encoder(x)
            x_hat = self.decoder(z)

            _loss, metrics = self.get_loss_and_metrics(
                [
                    x, x_hat, z, mean, std
                ]
            )

            outputs = {
                "loss": _loss,
                "metrics": metrics,
                "outputs": {
                    "scores": x_hat,
                    "x_pred": x_hat > 0.5,
                    "z": z,
                    "mean": mean,
                    "std": std
                }

            }
            return outputs

        if not self.is_distributed:
            ops = step(self.inputs.x)
            for key, value in ops["outputs"].items():
                setattr(self.outputs, key, value)

            self._losses.append(ops["loss"])
            self._metrics.extend(ops["metrics"])
        else:
            self._outs = dict()
            per_replica_dict = self.distributed_strategy.experimental_run_v2(
                step, args=(self.inputs.x, self.inputs.y)
            )
            loss = self.distributed_strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_dict["loss"]
            )
            self._outs["loss"] = loss

            for key, value in per_replica_dict["outputs"].items():
                v = self.distributed_strategy.experimental_local_results(value)
                setattr(self.outputs, key, tf.stack(v, axis=0, name="output.{}".format(key)))

            for value in per_replica_dict["metrics"]:
                m = self.distributed_strategy.reduce(
                    tf.distribute.ReduceOp.SUM, value
                )
                self._outs[m.name] = m

    def sample(self):
        z_shape = self.outputs.z.shape.as_list()[1:]
        z = tf.random_normal(shape=[self.batch_size] + z_shape)
        x_hat = self.decoder(z)
        self.outputs.x_hat = x_hat

    def get_loss_and_metrics(self, _inputs):
        """
        Loss definition.
        """
        [
            x, x_hat, z, mean, std
        ] = _inputs
        x_shape = x.shape.as_list()
        x_shape = x_shape[1:]

        # bce = tf.keras.losses.BinaryCrossentropy()
        # reconstruction_loss = bce(tf.keras.backend.flatten(x), tf.keras.backend.flatten(x_hat))
        reconstruction_loss = tf.keras.backend.binary_crossentropy(
            x,
            x_hat, 
            from_logits=False
        )
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[1,2,3])
        
        mse = tf.keras.metrics.mse(tf.keras.backend.flatten(x), tf.keras.backend.flatten(x_hat))
        mse *= np.prod(x_shape)

        # kl_loss = 1. + (2. * std) - tf.square(mean) - tf.exp(2. *std)
        # kl_loss = -.5 * tf.reduce_sum(kl_loss, axis=-1)
        # kl_loss = -0.5 * tf.reduce_sum(tf.exp(std) + tf.square(mean) - 1. - std, axis=-1)
        # kl_loss = -0.5 * tf.reduce_sum(tf.square(mean) + tf.square(std) - tf.log(1e-8 + tf.square(std)) - 1, [1])
        kl_loss = -0.5 * tf.reduce_sum(1 + std - tf.square(mean) - tf.exp(std), axis=-1)

        kl_loss = tf.identity(kl_loss, name="kl-divergence")
        _loss = tf.reduce_mean(reconstruction_loss + kl_loss, name="loss")
        
        print("######", _loss)
        # _loss = tf.reduce_mean(_loss, name="mean_binary_cross_entropy")

        return _loss, [tf.reduce_mean(mse, name="mse_"),tf.reduce_mean(kl_loss, name="KL"), tf.reduce_mean(reconstruction_loss, name="bce")]
        # return _loss, [tf.reduce_mean(std, name="std_mean"),tf.reduce_mean(mean, name="mean_sum") ,tf.reduce_mean(x_hat, name="scores_sum"),tf.reduce_mean(mse, name="mse_"),tf.reduce_mean(kl_loss, name="KL"), tf.reduce_mean(reconstruction_loss, name="bce")]

    def _callbacks(self):
        # from flow.callbacks.early_stop import EarlyStopping, ModeEnum
        # from flow.callbacks.checkpointer import CheckPointer, ModeEnum
        from flow.callbacks.checkpointer import CheckPointer, ModeEnum
        from flow.callbacks.history import History
        from flow.callbacks.timers import Timers
        # from flow.callbacks.before_epoch import BeforeEpoch

        self._checkpointer = CheckPointer(
            self.config.get("flow.checkpoint", "../data/models/"),
            monitor="bce",
            verbose=1,
            save_best_only=True,
            mode=ModeEnum.MIN,
            # max_to_keep=int(self.config.get("flow.max_checkpoints_to_keep", 3))
        )
        self.history = History(
            path=self.config.get("flow.checkpoint", "../data/models/"),
            add_keys=[
                # "valid_bce",
                # "valid_KL",
                "epoch_elapsed_time_in_seconds",
                "average_batch_elapsed_time_in_seconds",
                "learning_elapsed_time_in_seconds",

            ]
        )
        self._timers = Timers()

    def get_optimizer(self):
        # initial learning rate
        lr = tf.Variable(self.initial_lr, trainable=False)
        # from optim import Optimizer
        per_epoch_steps = len(self._train_dataset) // self.global_batch_size

        # Implements linear decay of the learning rate.
        if self.lr_decay_type == "polynomial_decay":
            lr = tf.train.polynomial_decay(
                learning_rate=lr,
                global_step=self.global_step,
                decay_steps=per_epoch_steps * self.decay_period_in_epochs,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False
            )
        # elif self. lr_decay_type == "linear_decay":
        #     decay_steps = per_epoch_steps * self.decay_period_in_epochs
        #     decay_rate = self.decay_rate

        if self.warmup_type == "linear":
            # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            warmup_steps = per_epoch_steps * self.warmup_steps_in_epochs
            global_step = tf.cast(self.global_step, dtype=tf.float32)

            warmup_percent = global_step / warmup_steps
            warmup_learning_rate = init_lr * warmup_percent
            is_warmup = tf.cast(global_step < warmup_steps, tf.float32)
            lr = (1.0 - is_warmup) * lr + (is_warmup * warmup_learning_rate)

        # optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        if self.gradient_clip:
            from tensorflow.contrib.estimator import clip_gradients_by_norm
            optimizer = clip_gradients_by_norm(
                optimizer, self.clip_norm
            )
        return optimizer

    def fit(
            self, train_dataset: Dataset, valid_dataset: Dataset=None,
            resume=False
    ):
        """
        Model fit function.
        :param train_dataset: training dataset partition.
        :param valid_dataset: valid dataset partition.
        :param resume: true if it is strating from an already run fitting state.
        """
        if self.is_distributed:
            self._distributed_fit(train_dataset, valid_dataset, resume)
        else:
            self._fit(train_dataset, valid_dataset, resume, learner=DefaultLearner)

    def _fit(
        self, train_dataset: Dataset,
        valid_dataset: Dataset = None,
        resume=False, learner=DefaultLearner
    ):
        # setting model iterator into dataset
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        if self._train_dataset is not None:
            self._train_dataset.set_iterator(self._iter)
        if self._valid_dataset is not None:
            self._valid_dataset.set_iterator(self._iter)

        if self._outs is None:
            outs = self._prepare_outputs(step="train")
            self._outs = outs

        if self._learner is None:
            self._learner = learner(self, self._outs, resume=resume)
        # TODO prepare inputs: it could be possible to pass the inputs as parameters.
        self._learner.fit()

    def _distributed_fit(
            self, train_dataset: Dataset, valid_dataset: Dataset = None, resume=False
    ):
        # some imports that are needed
        from flow.callbacks import before_session_initialization, on_validate_begin, on_epoch_begin
        from datasets.distributed_dataset import DistributedDataset
        from .learner import DistributedLearner
        # ensures train and valid datasets callbacks are disconected
        on_epoch_begin.disconnect(train_dataset.initialize_iterator)
        before_session_initialization.disconnect(train_dataset.get_iterator_initializer)
        if valid_dataset is not None:
            on_validate_begin.disconnect(valid_dataset.initialize_iterator)
            before_session_initialization.disconnect(valid_dataset.get_iterator_initializer)

        # saving the train and valid dataset reference
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

        # builds the distributed dataset wrapper
        distributed_ds = DistributedDataset(
            partitions={
                "train": train_dataset,
                "valid": valid_dataset
            },
            inputs_config=self.inputs_config,
            config=self.config,
            strategy=self.distributed_strategy
        )
        for name, tensor in zip(self.inputs_config["names"], distributed_ds.next_elements):
            setattr(self.inputs, name, tensor)

        with self.distributed_strategy.scope():
            self.get_model_spec()
            p = self.config.get("flow.premodel")
            self.load(tf.train.latest_checkpoint(p))
            sess = tf.get_default_session()
            # learner = DistributedLearner(self, self._outs, distributed_ds, resume, self.distributed_strategy)
            learner = DistributedLearner(self, self._outs, distributed_ds, False, self.distributed_strategy)
            learner.fit()

