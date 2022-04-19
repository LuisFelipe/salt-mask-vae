# -*- coding: utf-8 -*-
"""
module salt_dsv2.py
--------------------
salt dataset reader and iterator second version.
It reads the dataset from a preprocessed data path.
Additionally, it can be used to augment the data with horizontal flips.
"""
import os
import sys
import random
import warnings
import tensorflow as tf
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from PIL import Image
from flow.dataset import Dataset as DS

class SaltDS(DS):
    """Seismogram dataset iterator.
    Reads the dataset and iterates over it.
    """

    def __init__(self, inputs_config, config=None, path=None, partition="train", iterator=None, *args, **kwargs):
        """Main module initialization."""
        self._idx = None
        self._meta = None
        self._x = list()
        self._y = list()
        self.partition = partition
        self.path = path
        self.config = config
        self.random_state = np.random.RandomState(1234567)
        
        self._init_configs()
        # load img paths
        self.load_data()
        super().__init__(inputs_config, config, path, iterator, *args, **kwargs)

    def _init_configs(self):
        self.max_size = int(self.config.get("dataset.max_size", -1))
        self.flip_augment = self.config.get("dataset.flip_augment", "false").strip().lower() == "true"
        self.batch_size = int(self.config.get("flow.batch_size", 1))
        self.prefetch_buffer = int(self.config.get("DATASET.BUFFER_SIZE", 1))
        self.patch_size = [int(x) for x in self.config.get("dataset.patch_size").strip().lower().split(",")]

    def load_data(self):
        
        def load(path):
            # valid_size = float(self.config.get("dataset.valid_size", 0.1))
            for root, dirs, files in os.walk(path):
                for f in files:
                    if not f.endswith(".png"):
                        continue
                    if "_mask" in f:
                        continue
                    f_path = os.path.join(root, f)
                    self._x.append(f_path)
                    f_path = os.path.join(root, f.replace(".png", "_mask.png"))
                    self._y.append(f_path)
        
        if self.partition == "all":
            path = os.path.join(self.path, "train")
            load(path)
            path = os.path.join(self.path, "valid")
            load(path)
            path = os.path.join(self.path, "test")
            load(path)
            self.partition = "train"
        else:
            path = os.path.join(self.path, self.partition)
            load(path)
        

        self._y = np.asarray(self._y)
        self._x = np.asarray(self._x)

        # flip augmentation config
        self._meta = np.zeros_like(self._x, dtype=np.int8)
        if self.flip_augment:
            self._meta = np.concatenate(
                [
                    self._meta, 
                    np.ones_like(self._meta, dtype=np.int8),
                    np.ones_like(self._meta, dtype=np.int8) + 1,
                    np.ones_like(self._meta, dtype=np.int8) + 2,
                ],
                axis=0
            )
            self._x = np.tile(self._x, reps=4)
            self._y = np.tile(self._y, reps=4)

        # setting seed
        # deterministic shuffle ds
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]
        self._meta = self._meta[perm]

        # if dataset.max_size is set
        # then we truncate the dataset length to fit the max_size
        if self.max_size > 0:
            self._x = self._x[:self.max_size]
            self._y = self._y[:self.max_size]
            self._meta = self._meta[:self.max_size]

    def __len__(self):
        return len(self._x)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._x[item], self._y[item], self._meta[item]

        elif isinstance(item, slice):
            return self._x[item], self._y[item], self._meta[item]
        else:
            raise TypeError("Dataset indices must be integers or slices, not {}.".format(type(item)))

    def __iter__(self):
        self._idx = 0
        # random shuffle dataset on stop iteration
        perm = self.random_state.permutation(len(self._x))
        self._x = self._x[perm]
        self._y = self._y[perm]
        self._meta = self._meta[perm]
        return self

    def __next__(self):
        ds_len = len(self)
        
        mod_batch = ds_len % self.batch_size
        # stop iteration condition
        if self._idx >= (ds_len - mod_batch):
            raise StopIteration()
        # labels
        y = self.read_img(self._y[self._idx])
        
        # flip augmentation on demand
        y = self.augment(self._meta[self._idx], y)

        # go to next idx + 1
        self._idx += 1

        return (y,)

    def build_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self),
            output_types=self._inputs_config["output_types"],
            output_shapes=self._inputs_config["output_shapes"]
        )
        dataset = dataset.batch(self.batch_size)
        return dataset.prefetch(buffer_size=self.prefetch_buffer)

    def read_img(self, path):
        image = Image.open(path)
        if "_mask.png" in path:
            image = image.convert('L')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.float32)
            x = np.expand_dims(x, axis=-1)
            x = x / 255
        else:
            image = image.convert('RGB')
            image = image.resize((self.patch_size[1], self.patch_size[0]))
            x = np.asarray(image, dtype=np.float32)
            x /= 255
        
        return x

    def augment(self, augmentation_type, y):
        if augmentation_type == 1:
            y = np.fliplr(y)
        elif augmentation_type == 2:
            y = np.flipud(y)
        elif augmentation_type == 3:
            y = np.fliplr(y)
            y = np.flipud(y)

        return y
