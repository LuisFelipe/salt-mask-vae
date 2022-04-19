# -*- coding: utf-8 -*-
"""
module model.py
--------------------
Definition of the machine learning model for the task.
"""
import os
# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# setting visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Use tensorcores
# enable tf mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL) 
tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel("ERROR")
import numpy as np
from flow.flow_main import FlowMain
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import utils
sns.set_style("white")
plt.style.use('seaborn-white')


class Main(FlowMain):
    __config_path__ = ""

    def __init__(self, session=None, config_name=None, config_path=None):
        # tf.compat.v1.random.set_random_seed(54321)
        # np.random.seed(4325)
        super().__init__(session=session, config_name=config_name, config_path=config_path)

        resolution = [int(x) for x in self.config.get("dataset.patch_size", "64,64").strip().split(",")]
        h_resolution = resolution[0]
        w_resolution = resolution[1]
        # dataset tensor specifications
        self.inputs_config = {
            "output_types": (
               tf.float32,  # x
            ),
            "output_shapes": (
                tf.TensorShape([h_resolution, w_resolution, 1]),  # x
            ),
            "names": (
                "x",
            )
        }
        self.model_name = self.config.get("model.name", "vae")
        self.ds_name = self.config.get("dataset.dataset_name", "salt_ds")

    def get_model(self, *args, **kwargs): 
        if self.model_name == "vae":
            from models.vae import Vae
            model = Vae(
                # ensuring compatibility with old version of flow. to be depricated
                inputs_config=self.inputs_config,
                config=self.config
            )
            return model
        elif self.model_name == "vaeV2":
            from models.vae_v2 import Vae
            model = Vae(
                # ensuring compatibility with old version of flow. to be depricated
                inputs_config=self.inputs_config,
                config=self.config
            )
            return model
        elif self.model_name == "vaeV3":
            from models.vae_v3 import Vae
            model = Vae(
                # ensuring compatibility with old version of flow. to be depricated
                inputs_config=self.inputs_config,
                config=self.config
            )
            return model
        else:
            raise ValueError(
                "Model name not recognizes."
                " Received {}. Allowed values are: \"vae\", \"vaeV2\"."
            )

    def get_dataset(self):

        if self.ds_name == "salt_ds":
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
            return train_ds, valid_ds, #test_ds
        elif self.ds_name == "salt_dsV2":
            from datasets.salt_dsv2 import SaltDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="all"
            )
            valid_ds = SaltDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            # test_ds = SaltDS(
            #     self.inputs_config,
            #     config=self.config,
            #     path=ds_path,
            #     partition="test"
            # )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds, #test_ds
        elif self.ds_name == "mnist":
            from datasets.mnist import MnistDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = MnistDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="train"
            )
            valid_ds = MnistDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="valid"
            )
            # test_ds = MnistDS(
            #     self.inputs_config,
            #     config=self.config,
            #     path=ds_path,
            #     partition="test"
            # )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds
        elif self.ds_name == "kaggle_ds":
            from datasets.kaggle_aug_ds import KaggleAugDS
            ds_path = self.config.get("dataset.path", "")
            train_ds = KaggleAugDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="train",
            )
            valid_ds = KaggleAugDS(
                self.inputs_config,
                config=self.config,
                path=ds_path,
                partition="valid",
            )
            print(">>>>>ds_len>>>>>", len(train_ds))
            return train_ds, valid_ds, #test_ds

    def train(self, *args, **kwargs):

        should_resume = self.config.get("flow.resume", "False").lower() == "true"
        train_ds, valid_ds = self.get_dataset()
        model = self.get_model()
        if should_resume:
            p = self.config.get("flow.premodel")
            model.load(tf.train.latest_checkpoint(p))

        model.fit(
            train_dataset=train_ds,
            # valid_dataset=valid_ds,
            # resume=should_resume
        )
        return model

    def predict(self, *args, **kwargs):
        train_ds, valid_ds = self.get_dataset()
        model = self.get_model()
        model.sample()
        p = self.config.get("flow.premodel")
        model.load(tf.train.latest_checkpoint(p))
        model._is_session_initialized = True

        for outs in model.predict(train_ds, {"x_hat": model.outputs.x_hat}):
            x = outs["x_hat"]
            for i in range(10):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                xaux = np.squeeze(x[i] > .5)
                xaux = utils.mask_post_process(xaux)
                ax.imshow(xaux, cmap="gray")
                # ax2.imshow(xaux, cmap="gray")
                # ax.imshow(np.squeeze(x[i]), cmap="gray")
                plt.show()
            break

    def sample(self, *args, **kwargs):
        """Sample and save masks."""
        train_ds, valid_ds = self.get_dataset()
        model = self.get_model()
        model.sample()
        p = self.config.get("flow.premodel")
        model.load(tf.train.latest_checkpoint(p))
        model._is_session_initialized = True

        # sample configs
        n_samples = int(self.config.get("samples.n_samples"))
        save_path = self.config.get("samples.save_path")
        threshold = float(self.config.get("samples.thresold", "0.5"))
        utils.mkdir_if_not_exists(save_path)
        count = 0

        for outs in model.predict(train_ds, {"x_hat": model.outputs.x_hat}):
            x = outs["x_hat"]
            for i in range(len(x)):
                xaux = np.squeeze(x[i] > threshold)
                # xaux = x[i]
                xaux = utils.mask_post_process(xaux)
                # np.save(
                #     file=os.path.join(save_path, "{}.npy".format(count)),
                #     arr=xaux
                # )
                utils.save_figure_from_array(
                    path=os.path.join(save_path, "{}.png".format(count)), 
                    array=xaux
                )

                # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                # ax.imshow(np.squeeze(xaux), cmap="gray")
                # plt.show()

                count += 1
                if count >= n_samples:
                    break

            if count >= n_samples:
                break

    def evaluate(self):
        # evaluation is made in a non-distributed setting
        self.config["flow.distributed"] = "false"
        # loading data
        model = self.get_model()
        train_ds, _ = self.get_dataset()
        p = self.config.get("flow.premodel")
        model.load(tf.train.latest_checkpoint(p))
        model._is_session_initialized = True
        evaluations = model.evaluate(train_ds)
        print("**** metrics ****", evaluations)
        # outs = model.evaluate(ds)
        print(">>>model_evals>>>")

    def print(self):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-white')
        import seaborn as sns
        sns.set_style("white")
        from PIL import Image
        
        # loading data
        train_ds, valid_ds,  = self.get_dataset()
        print("**********", len(train_ds), len(valid_ds))
        # model = self.get_model()

        train_iter = iter(train_ds)
        
        for i in range(10):
            x, = next(train_iter)
            print("*******", x.shape, x.dtype)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(np.squeeze(x), cmap="gray")
            plt.show()

    def print_history(self):
        hist_path = self.config.get("flow.checkpoint")
        hist_path = os.path.join(hist_path, "_history.json")

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-white')
        import seaborn as sns
        sns.set_style("white")
        import json

        with open(hist_path, mode="r") as fp:
            _json = json.load(fp)
        iou = _json["IoU"]
        valid_iou = _json["valid_IoU"]
        # elapsed_time = _json["epoch_elapsed_time_in_seconds"][1:]
        # print("elapsed_time: ", sum(elapsed_time)/60, (sum(elapsed_time)/60)/60)

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.set(xlabel="Epoch", ylabel="IoU")
        ax.plot(iou, label="train iou")
        ax.plot(valid_iou, label="valid iou")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.__enter__()
    # sess_ctx = session.as_default()
    # sess_ctx.__enter__()
    main = Main(tf.get_default_session())
    # main.train()
    # main.print()
    # main.predict()
    main.sample()
