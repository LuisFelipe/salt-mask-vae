# -*- coding: utf-8 -*-
"""
module dropout.py
-----------------------------
A dropout layer implementation.
"""
import tensorflow as tf
from flow.layers.flow_layer import FlowLayer
layers = tf.keras.layers
from tensorflow.contrib import autograph

# _dropout = autograph.do_not_convert()(tf.nn.dropout)

@tf.function
def dropout(learning, x, rate, seed):
    if learning:
        return tf.nn.dropout(
            x=x,
            rate=rate,
            noise_shape=x.shape,
            seed=seed
        )
    else:
        return x


class Dropout(FlowLayer):
    """
    Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    """

    def __init__(self, rate, seed=None, name=None, **kwargs):
        """
        Dropout Layer initialization.
        :param rate: float between 0 and 1. Fraction of the input units to drop.
        :param seed: A Python integer to use as random seed.
        """
        self.rate = rate
        self.seed = seed
        self.supports_masking = True
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        """
        Builds the tensorflow layer's graph.
        :param input_shape: object representing the tensor shape.
        :type input_shape: tf.TensorShape
        """
        if self.built:
            return

        super().build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):

        with tf.variable_scope("learning_phase", reuse=tf.AUTO_REUSE):
            training = tf.get_variable(name="is_training_phase", dtype=tf.bool, initializer=False)

        return dropout(training, inputs, self.rate, self.seed)
