# -*- coding: utf-8 -*-
"""
module GridSearch.py
--------------------
Grid Search Procedures.
"""
import os
import numpy as np
import json
import tensorflow as tf
from flow.numpy_encoder import NumpyEncoder
from datetime import datetime

def clear_callbacks():
    from flow.callbacks import on_epoch_begin, on_epoch_end, on_batch_begin, on_batch_end, on_train_begin, on_train_end, validate_sig, on_validate_begin, on_validate_end, before_session_initialization
    on_epoch_begin._clear_state()
    on_epoch_end._clear_state()
    on_batch_begin._clear_state()
    on_batch_end._clear_state()
    on_train_begin._clear_state()
    on_train_end._clear_state()
    validate_sig._clear_state()
    on_validate_begin._clear_state()
    on_validate_end._clear_state()
    before_session_initialization._clear_state()

class BaseGridSearch(object):
    """ Base Grid Seach Class.
    """
    def __init__(self, inputs_config, config, start=8, end=64, step=2):
        """ Grid initializer.

        [description]
        :param datasets: train, valid and test datasets.
        :param start: minimum number os augmentations, defaults to 0
        :type start: int, optional
        :param end: maximum number of augmentations, defaults to 1000
        :type end: int, optional
        :param step: step size, defaults to 100
        :type step: int, optional
        """ 
        self.start = start
        self.end = end
        self.step = step
        self.inputs_config = inputs_config
        self.config = config

    def get_model(self):
        from models.unet import Unet
        model = Unet(
            # ensuring compatibility with old version of flow. to be depricated
            inputs_config=self.inputs_config,
            config=self.config
        )
        return model

    def get_dataset(self):
        ds_name = self.config.get("dataset.dataset_name", "salt_ds")
        if ds_name == "salt_ds":
            from datasets.salt_ds import SaltDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                model_path=self.config.get("dataset.train.model_path"),
                velocity_path=self.config.get("dataset.train.velocity_path"),
                model_size=[int(x) for x in self.config.get("dataset.train.model_size", "1040,7760").strip().split(",")],
                strip=[int(x) for x in self.config.get("dataset.train.strip").strip().split(",")],
                mask_threshold=float(self.config.get("dataset.train.mask_threshold")),
                patch_size=[int(x) for x in self.config.get("dataset.patch_size", "64,64").strip().split(",")],
                partition="train",
                partition_bound=int(self.config.get("dataset.train.partition_bound").strip())
            )
            valid_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                model_path=self.config.get("dataset.valid.model_path"),
                velocity_path=self.config.get("dataset.valid.velocity_path"),
                model_size=[int(x) for x in self.config.get("dataset.valid.model_size", "1040,7760").strip().split(",")],
                strip=[int(x) for x in self.config.get("dataset.valid.strip").strip().split(",")],
                mask_threshold=float(self.config.get("dataset.valid.mask_threshold")),
                patch_size=[int(x) for x in self.config.get("dataset.patch_size", "64,64").strip().split(",")],
                partition="valid",
                partition_bound=int(self.config.get("dataset.valid.partition_bound").strip())
            )
            test_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                model_path=self.config.get("dataset.test.model_path"),
                velocity_path=self.config.get("dataset.test.velocity_path"),
                model_size=[int(x) for x in self.config.get("dataset.test.model_size", "6912,1216").strip().split(",")],
                strip=[int(x) for x in self.config.get("dataset.test.strip").strip().split(",")],
                mask_threshold=float(self.config.get("dataset.test.mask_threshold")),
                patch_size=[int(x) for x in self.config.get("dataset.patch_size", "64,64").strip().split(",")],
                partition="test",
                partition_bound=int(self.config.get("dataset.test.partition_bound").strip())
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds, test_ds
        elif ds_name == "salt_dsV2":
            from datasets.salt_dsv2 import SaltDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="train"
            )
            valid_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            test_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="test"
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds, test_ds

    def predict(self, model, dataset):
        p = self.config.get("flow.premodel")
        model.load(tf.train.latest_checkpoint(p))
        print("evaluating...")
        y_pred = list()
        y_true = list()
        for ret in model.predict(dataset, outputs={"y_true": model.inputs.y, "y_pred": model.outputs.y_pred}):
            y_pred.append(ret["y_pred"])
            y_true.append(ret["y_true"])

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        return y_true, y_pred

    def get_metrics(self, y_true, y_pred):
        from metrics import f1_metric, precision_smooth_np, recall_smooth_np, iou_smooth_np
        from sklearn.metrics import confusion_matrix, accuracy_score

        prec = precision_smooth_np(y_true, y_pred)
        recall = recall_smooth_np(y_true, y_pred)
        f1 = f1_metric(prec, recall)
        iou = iou_smooth_np(y_true, y_pred)

        print('---------------------------------------------------------')
        print('F1 score per class (good,bad,ugly): ', f1)
        print('Precision per class: ', prec)
        print('Recall per class: ', recall)
        print('IoU: ', iou)
        print('---------------------------------------------------------')

        return f1, prec, recall, iou
    
    def save_results(self, results):
        checkpoint = self.config["flow.CHECKPOINT"]
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        now = datetime.now().strftime("%y-%m-%d-%H-%M")
        with open(checkpoint + "{}_{}_results.json".format(now, str(self.__class__.__name__)), mode="w", encoding='utf-8') as f:
            json.dump(results, f, cls=NumpyEncoder)


class AlphaGridSearch(BaseGridSearch):
    """ Grid search on the number of filters of the unet model.
    """
    def __init__(self, inputs_config, config, start=8, end=64, step=2):
        """ Grid initializer.

        :param datasets: train, valid and test datasets.
        :param start: minimum number os augmentations, defaults to 0
        :type start: int, optional
        :param max_augs: maximum number of augmentations, defaults to 1000
        :type max_augs: int, optional
        :param step: step size, defaults to 100
        :type max_augs: int, optional
        """ 
        super().__init__(inputs_config, config, start=8, end=64, step=2)

    def execute(self):
        from tensorflow.python.framework import ops
        from tensorflow.keras.backend import reset_uids
        # reinitialize_op = reinitialize_all_variables()
        sess = tf.get_default_session()
        current_size = self.start

        f1s = list()
        precs = list()
        recalls = list()
        predictions = list()
        ious = list()
        epochs = list()
        iterations = list()
        alphas = list()
        while current_size <= self.end:
            sess = tf.get_default_session()
            sess.__exit__(None, None, None)
            sess.close()
            clear_callbacks()

            sess = tf.Session()
            tf.reset_default_graph()
            sess.__enter__()
            tf.keras.backend.set_session(tf.get_default_session())
            self.config["model.alpha"] = current_size
            print(">>>fitting model>>>> alpha = {} ....".format(current_size))

            train_ds, valid_ds, test_ds = self.get_dataset()
            model = self.get_model()
            test_ds.set_iterator(model._iter)
            model.fit(
                train_dataset=train_ds,
                valid_dataset=valid_ds,
            )

            y_true, y_pred = self.predict(model, test_ds)
            f1, prec, recall, iou = self.get_metrics(y_true, y_pred)

            f1s.append(f1)
            precs.append(prec)
            recalls.append(recall)
            ious.append(iou)
            predictions.append({"y_true": y_true, "y_pred": y_pred})
            # epochs.append(model._early_stop.best_epoch)
            # iterations.append(model._early_stop.best_iteration)
            alphas.append(current_size)
            current_size *= self.step
        
        print("saving_results....")

        results = {
            "f1": f1s,
            "precision": precs,
            "recall": recalls,
            "iou": ious,
            "predictions": predictions,
            # "epochs": epochs,
            # "iterations": iterations,
            "alphas": alphas,
            "start": self.start,
            # "max_augs": self.max_augs,
            "step_size": self.step
        }
        self.save_results(results)



