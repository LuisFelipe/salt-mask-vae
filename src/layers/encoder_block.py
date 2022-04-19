# -*- coding: utf-8 -*-
"""
module conv2d_block.py
-------------------------
    Conv2d_zeros layers used in the prior calculation.
    Code adapted from the official openAI ´Glow´{https://github.com/openai/glow} repository.
"""
import tensorflow as tf
import numpy as np
from flow.layers import FlowLayer
from layers.dropout import Dropout


class EncoderBlockV2(FlowLayer):
    """Unet Conv2d_block.
    """

    def __init__(
            self, filters=64, activation="relu", kernel_initializer="he_normal", n_latent=8,
            name=None, **kwargs
    ):
        """ Unet Conv2D Block initializer.
        """
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.n_latent = n_latent
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        self._x_shape = input_shape

        self._blocks = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
                # kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
                # kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Dense(
                units=self.filters,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer
                # kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Dense(
                units=self.n_latent*2,
                activation=None,
                use_bias=True,
                # kernel_initializer=self.kernel_initializer
                kernel_initializer=tf.initializers.zeros()
            )
        ]

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        for l in self._blocks:
            x = l(x)

        mu = x[:, :self.n_latent]
        sigma = x[:, self.n_latent:]

        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
        # z  = mean + epsilon * tf.sqrt(tf.exp(std))
        z = mu + epsilon * tf.sqrt(tf.exp(sigma))
        # return z, mean, std
        return z, mu, sigma


class EncoderBlock(FlowLayer):
    """Unet Conv2d_block.
    """

    def __init__(
            self, filters=64, kernel_size=[4, 4], strides=[2, 2],
            dropout_rate=0.3, activation="relu",
            padding="SAME", kernel_initializer="he_normal", n_latent=8,
            name=None, **kwargs
    ):
        """ Unet Conv2D Block initializer.
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
        self._blocks = list()
        
        self._blocks.append(
            tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                padding=self.padding 
            )
        )
        
        if self.dropout_rate > 0.:
            self._blocks.append(
                Dropout(rate=self.dropout_rate)
            )

        self._blocks.append(
            tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                padding=self.padding 
            )
        )

        if self.dropout_rate > 0.:
            self._blocks.append(
                Dropout(rate=self.dropout_rate)
            )

        self._blocks.append(
            tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                padding=self.padding 
            )
        )
        
        if self.dropout_rate > 0.:
            self._blocks.append(
                Dropout(rate=self.dropout_rate)
            )
        

        self._blocks.append(
            tf.keras.layers.Flatten()
        )

        self.mu_and_sigma = tf.keras.layers.Dense(
            units=self.n_latent*2,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=self.kernel_initializer
            # kernel_initializer=tf.initializers.zeros()
        )

        self.scale = tf.get_variable(
                name=self.name + ".scale",
                shape=(1, self.n_latent*2),
                dtype=tf.float32,
                initializer=tf.ones_initializer()
            )

        # self.std = [
        #     tf.keras.layers.Dense(
        #         units=self.n_latent,
        #         # activation=tf.nn.softplus,
        #         activation=None,
        #         use_bias=True,
        #         kernel_initializer=self.kernel_initializer
        #     )
        # ]

        super().build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        for l in self._blocks:
            x = l(x)
        
        # mean = self.mean(x)
        mu_and_sigma = self.mu_and_sigma(x) * self.scale
        mu = mu_and_sigma[:, :self.n_latent]
        sigma = mu_and_sigma[:, self.n_latent:]
        # std = x
        # for l in self.std:
        #     std = l(std)

        # sigma += 1e-5
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
        # z  = mean + epsilon * tf.sqrt(tf.exp(std))
        z = mu + epsilon * tf.sqrt(tf.exp(sigma))
        # return z, mean, std
        return z, mu, sigma
