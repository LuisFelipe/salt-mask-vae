# -*- coding: utf-8 -*-
"""
module conv2d.py
-----------------------------
Convolutional Neural network blocks used inside the image_transformer model.
"""
# Copyright 2018 Luis Felipe Müller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from flow.layers.flow_layer import FlowLayer
import numpy as np
layers = tf.keras.layers


class Conv1x1(FlowLayer):
    """
    1x1 Convolutional 2D Layer.
    """
    def __init__(self, filters, name=None, **kwargs):
        self.filters = filters
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        """
        Builds the tensorflow layer's graph.
        :param input_shape: object representing the tensor shape.
        :type input_shape: tf.TensorShape
        """
        if self.built:
            return

        with tf.variable_scope(self.name):
            x_shape = input_shape.as_list()
            self.kernel_width = 1
            self.kernel_height = 1
            inputs_dim = x_shape[-1]

            self.w = self.add_weight(
                name="w",
                shape=[self.kernel_height, self.kernel_width, inputs_dim, self.filters],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True
            )

            # self.weight_scale = self.add_weight(name="weight_scale", shape=[], dtype=tf.float32, initializer=tf.initializers.ones())
            # self.weight_shift = self.add_weight(name="weight_shift", shape=[], dtype=tf.float32, initializer=tf.initializers.zeros())
            super().build(input_shape)
            self.built = True

    def call(self, inputs, training=None, mask=None):
        """
        Execute input layer.
        :param inputs: Layers input Tensor.
        :param training: True when running in the training step.
        :param mask: a mask for the inputs.
        :return: the layers output tensor.
        """
        # w = self.norm_weights(self.w)
        conv = tf.nn.conv2d(
            input=inputs,
            filter=self.w,
            strides=[1, self.kernel_height, self.kernel_width, 1],
            padding='SAME'
        )
        return conv


class Conv2D(FlowLayer):
    """
    Convolutional 2D Layer.
    """

    def __init__(self, filters, kernel_size, strides, padding, add_bias=False, name=None, **kwargs):
        self.nfilters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.lower()
        self.add_bias= add_bias
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        """
        Builds the tensorflow layer's graph.
        :param input_shape: object representing the tensor shape.
        :type input_shape: tf.TensorShape
        """
        if self.built:
            return

        with tf.variable_scope(self.name):
            x_shape = input_shape.as_list()
            self.kernel_width = self.kernel_size[1]
            self.kernel_height = self.kernel_size[0]
            inputs_dim = x_shape[-1]

            self.w = self.add_weight(
                name="w",
                shape=[self.kernel_height, self.kernel_width, inputs_dim, self.nfilters],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            self.b = self.add_weight(
                name='biases',
                shape=[self.nfilters],
                initializer=tf.zeros_initializer()
            )

            # self.weight_scale = self.add_weight(name="weight_scale", shape=[], dtype=tf.float32,
            #                                     initializer=tf.initializers.ones())
            # self.weight_shift = self.add_weight(name="weight_shift", shape=[], dtype=tf.float32,
            #                                     initializer=tf.initializers.zeros())
            super().build(input_shape)
            self.built = True

    def call(self, inputs, training=None, mask=None):
        """
        Execute input layer.
        :param inputs: Layers input Tensor.
        :param training: True when running in the training step.
        :param mask: a mask for the inputs.
        :return: the layers output tensor.
        """
        # w = self.norm_weights(self.w)
        # b = self.norm_weights(self.b)
        conv = tf.nn.conv2d(
            input=inputs,
            filter=self.w,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding='SAME'
        )
        if self.add_bias:
            conv = tf.nn.bias_add(conv, self.b)
        return conv

    def norm_weights(self, weights):
        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(weights)
            ), name=self.name + "_norm"
        )
        mean = tf.reduce_mean(weights)
        # w = (weights - mean) / norm
        w = (weights) / norm
        # w *= self.weight_scale
        # w += self.weight_shift
        return w
