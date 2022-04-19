from invoke import task
from main import Main
import os
import json
import tensorflow as tf
import sys

sys.path.append('..')

@task
def train(ctx, data=None, config=''):
    """
    Function to Train the Classifier.

    Receives the dataset path and a classification file, train the model over the labels presented
    in the classification file.

    :param data: path to the desired data folder
    :param labels:  path to the classification file.
        The classification file contains the informations from the filename and class to each sismogram.
    :return: save the predictions in a given file
    """
    # Set GPU device, Limit GPU's Memory and Growth of GPU's memory
    # default_config = tf.ConfigProto(log_device_placement=self.config.log_device_placement)
    # default_config.gpu_options.visible_device_list = self.config.gpu

    # default_config.gpu_options.per_process_gpu_memory_fraction = self.config.gpu_memory
    # default_config.gpu_options.allow_growth = self.config.allow_growth
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    cfg_name = config.split("/")[-1]
    cfg_dir = config.replace(cfg_name, "")

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.__enter__()

    main = Main(tf.get_default_session(), config_name=cfg_name, config_path=cfg_dir)
    if data is not None:
        main.config["dataset.path"] = data

    main.train()


@task
def sample(ctx, config, save_path=None, n_samples=None):
    """
    Function to Predict Seismogram Class

    Receives a given data_folder and save the results in a given file.

    :param data: abs path to the desired data folder
    :param save_file: filename path to save the predictions(*.csv)
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    cfg_name = config.split("/")[-1]
    cfg_dir = config.replace(cfg_name, "")

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.__enter__()

    main = Main(tf.get_default_session(), config_name=cfg_name, config_path=cfg_dir)

    if save_path is not None:
        main.config["samples.save_path"] = save_path
    if n_samples is not None:
        main.config["samples.n_samples"] = n_samples

    # main.predict()
    main.sample()

@task
def datalen(ctx, config, save_path=None, n_samples=None):
    """
    Function to Predict Seismogram Class

    Receives a given data_folder and save the results in a given file.

    :param data: abs path to the desired data folder
    :param save_file: filename path to save the predictions(*.csv)
    :return: save the predictions in a given file
    """
    if not os.path.exists(config):
        raise FileNotFoundError(
            "The configuration file \"{}\" was not found.\n "
            "Please check if the path is correct or if the files does exists."
        )

    cfg_name = config.split("/")[-1]
    cfg_dir = config.replace(cfg_name, "")

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.__enter__()

    main = Main(tf.get_default_session(), config_name=cfg_name, config_path=cfg_dir)
    train_ds, valid_ds = main.get_dataset()
    print(">>train_len>>", len(train_ds))
    print(">>valid_len>>", len(valid_ds))
