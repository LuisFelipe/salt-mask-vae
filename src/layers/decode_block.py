# -*- coding: utf-8 -*-
"""
module decoder_block.py
-------------------------
    decoder blocl layers.
"""
import tensorflow as tf
import numpy as np
from flow.layers import FlowLayer
from layers.dropout import Dropout


class DecoderBlockV2(FlowLayer):
    """Unet Conv2d_block.
    """

    def __init__(
            self, filters=16, activation="relu", kernel_initializer="he_normal",
            name=None, **kwargs
    ):
        """ Unet Encode Block initializer.
        """
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        x_orig = tf.get_collection("flow_inputs")[0]
        x_orig = x_orig.shape.as_list()
        reshaped_dim = [-1, x_orig[1], x_orig[2], 1]

        self._blocks = [
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            ),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            ),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            ),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            ),
            # tf.keras.layers.Dense(
            #     units=self.filters,
            #     activation=self.activation,
            #     use_bias=True,
            #     kernel_initializer=self.kernel_initializer
            # ),
            tf.keras.layers.Dense(
                units=x_orig[1] * x_orig[2],
                activation=tf.nn.sigmoid,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            ),
            lambda x: tf.reshape(x, shape=reshaped_dim),

        ]

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        for l in self._blocks:
            x = l(x)
        return x


class DecoderBlock(FlowLayer):
    """Unet Conv2d_block.
    """

    def __init__(
            self, filters=16, kernel_size=[3, 3], strides=[1, 1],
            dropout_rate=0.3, activation="relu",
            padding="SAME", kernel_initializer="he_normal", n_latent=49,
            name=None, **kwargs
    ):
        """ Unet Encode Block initializer.
        """
        self.filters = filters
        self.kernel_size = list(kernel_size)
        self.strides = list(strides)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.padding = padding.upper()
        self.kernel_initializer = kernel_initializer
        self.n_latent = n_latent
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        reshaped_dim = [-1, self.n_latent, self.n_latent, 1]
        
        self._blocks = list()
        
        self._blocks.append(
            tf.keras.layers.Dense(
                units=self.n_latent**2,
                activation=None,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            )
        )
        self._blocks.append(
            tf.keras.layers.Dense(
                units=self.n_latent**2,
                activation=None,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            )
        )
        self._blocks.append(
            lambda x: tf.reshape(x, reshaped_dim)
        )

        self._blocks.append(
            tf.keras.layers.Conv2DTranspose(
                filters=self.filters, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                activation=self.activation, 
                padding=self.padding, 
                kernel_initializer=self.kernel_initializer, 
                name=self.name + ".Conv2DTranspose"
            )
        )

        if self.dropout_rate > 0.:
            self._blocks.append(
                Dropout(rate=self.dropout_rate)
            )

        # self._blocks.append(
        #     tf.keras.layers.Conv2D(
        #         filters=self.filters, 
        #         kernel_size=self.kernel_size, 
        #         strides=(1,1), 
        #         activation=self.activation, 
        #         padding=self.padding, 
        #         kernel_initializer=self.kernel_initializer, 
        #         name=self.name + ".Conv2DTranspose2"
        #     )
        # )

        # if self.dropout_rate > 0.:
        #     self._blocks.append(
        #         Dropout(rate=self.dropout_rate)
        #     )

        self._blocks.append(
            tf.keras.layers.Conv2D(
                filters=self.filters, 
                kernel_size=self.kernel_size, 
                strides=(1,1), 
                activation=self.activation, 
                padding=self.padding, 
                kernel_initializer=self.kernel_initializer, 
                name=self.name + ".Conv2DTranspose3"
            )
        )

        self._blocks.append(
            tf.keras.layers.Flatten()
        )

        self._blocks.append(
            tf.keras.layers.Dense(
                units=64*64,
                # units=28*28,
                activation=tf.nn.sigmoid,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
            )
        )
        
        self._blocks.append(
            # lambda x: tf.reshape(x, shape=[-1, 28, 28, 1])
            lambda x: tf.reshape(x, shape=[-1, 64, 64, 1])
        )

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        for l in self._blocks:
            x = l(x)
        return x