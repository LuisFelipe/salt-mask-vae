# -*- coding: utf-8 -*-
"""
module self_attention.py
-----------------------------
SelfAttention Neural network blocks used inside the image_transformer model.
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
from .conv2d import Conv2D, Conv1x1
from .layer_normalization import LayerNormalization
from .dropout import Dropout
layers = tf.keras.layers


class SelfAttention(FlowLayer):
    """
    Image Transformer multihead Local Self-Attention Layer
    """

    def __init__(
            self, flange_size, qk_units, v_units,
            num_heads, dropout_rate=-1.0, use_sigma=True,
            normalize=True, sigma=-2.0, out_units=None, skip_res=False,
            name=None, **kwargs
    ):
        self.flange_size = flange_size
        self.qk_units = qk_units
        self.v_units = v_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        # true when layer should use sigma multiplier
        self.use_sigma = use_sigma
        # initial sigma value
        self.initial_sigma = sigma
        self.normalize = normalize
        self.out_units = out_units
        self.skip_res = skip_res
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.built:
            return

        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 3:
                x_shape, orig, mask_shape = input_shape
            else:
                x_shape, orig = input_shape
        else:
            x_shape = input_shape
            orig = input_shape

        self.q_conv = layers.Conv2D(
            filters=self.qk_units * self.num_heads,
            kernel_size=self.flange_size,
            # strides=self.block_size,
            strides=(1,1),
            padding="same",
            use_bias=False,
            activation=None
        )

        self.k_conv = layers.Conv2D(
            filters=self.qk_units * self.num_heads,
            # strides=self.block_size,
            strides=(1,1),
            kernel_size=self.flange_size,
            padding="same",
            use_bias=False,
            activation=None
        )

        self.v_conv = layers.Conv2D(
            filters=self.v_units * self.num_heads,
            # strides=self.block_size,
            strides=(1,1),
            kernel_size=self.flange_size,
            padding="same",
            use_bias=False,
            activation=None
        )

        if self.use_sigma:
            self.sigma = self.add_weight(
                name='sigma_ratio',
                shape=[],
                initializer=tf.constant_initializer(self.initial_sigma),
                trainable=True
            )
        if self.out_units is not None:
            self.attention_deconv = Conv1x1(filters=self.out_units)
        else:
            self.attention_deconv = Conv1x1(filters=orig.as_list()[-1])

        if self.dropout_rate > 0:
            self.dropout = Dropout(self.dropout_rate, name=self.name + ".att_dropout")
            self.out_dropout = Dropout(self.dropout_rate, name=self.name + ".dropout")

        if self.normalize:
            self.layer_norm = LayerNormalization()

        super().build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        with tf.variable_scope(self.name):
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 2:
                    x, orig = inputs
                    mask = None
                else:
                    x, orig, mask = inputs
                    x_shape = x.shape.as_list()
                    mask = tf.reshape(mask, shape=[x_shape[0], 1, 1, -1])
            else:
                x = inputs
                orig = x
                mask = None

            q = self.q_conv(x)
            k = self.k_conv(x)
            v = self.v_conv(x)

            # reshaping qkv to match the flowing shape:
            # [batch, n_blocks, num_heads, HW/4, channels / num_heads]
            q_headed = self.reshape_and_split_heads(q)
            k_headed = self.reshape_and_split_heads(k)
            v_headed = self.reshape_and_split_heads(v)
            # attention bias calculation

            ########################
            # attention calculation
            ########################
            attn_output = self.dot_product_attention([q_headed, k_headed, v_headed], masks=mask)
            # output_shape = [n_batch, n_heads, w * h, v_channels]

            x_shape = x.shape.as_list()
            attn_output = self.reshape_and_concat_heads(attn_output, x_shape)
            # last output transform op
            # mix heads and put it back to the original shape
            attn_deconv = self.attention_deconv(attn_output)
            # TODO dropout attn_deconv
            if self.dropout_rate > 0:
                attn_deconv = self.out_dropout(attn_deconv)
            
            if not self.skip_res:
                # residual connection with attention inputs
                # if it should learn to weight the attention (should use gamma)
                if self.use_sigma:
                    # then multiply the attn_deconv by sigma
                    # before the residual connection
                    output = (attn_deconv * tf.nn.sigmoid(self.sigma)) + orig
                    if self.normalize:
                        output = self.layer_norm(output)
                else:
                    # otherwise, just adds the layer's input to the attended output
                    output = attn_deconv + orig
                    if self.normalize:
                        output = self.layer_norm(output)
            else:
                output = attn_deconv

            return output

    def reshape_and_concat_heads(self, t, x_shape):
        t = tf.transpose(t, perm=[0,2,1,3])
        t = tf.reshape(
            t,
            shape=[
                x_shape[0],
                x_shape[1],
                x_shape[2],
                self.v_units * self.num_heads
            ]
        )
        return t

    def reshape_and_split_heads(self, x):
        """Split channels (dimension 3) into multiple heads (becomes dimension 1).

        :param x: a Tensor with shape [batch, height, width, channels]
        :param num_heads: an integer

        :return: a Tensor with shape [batch, num_heads, height* width, channels / num_heads]
        """
        # puts x in shape = [batch, num_heads, height, width, channels / num_heads]
        x_shape = x.shape.as_list()
        out = tf.reshape(
            x,
            shape=[
                x_shape[0],
                int(x_shape[1]*x_shape[2]),
                self.num_heads,
                x_shape[3]//self.num_heads
            ]
        )
        out = tf.transpose(out, perm=[0,2,1,3])
        return out

    def dot_product_attention(self, qkv, masks=None):
        """Dot-product attention.

        :param q: Tensor with shape [..., length_q, depth_k].
        :param k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
          match with q.
        :param v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
          match with q.
        :param bias: bias Tensor.
        bias is used to remove attention from padded units

        :return: Tensor with shape [..., length_q, depth_v].
        """
        [q, k, v] = qkv
        # attention q dot k
        logits = tf.matmul(q, k, transpose_b=True)
        # logits_shape = logits.shape.as_list()
        # logits /= np.sqrt(logits_shape[-1])
        logits /= np.sqrt(self.v_units)

        if masks is not None:
            logits -= (1 - masks)*1e25

        # atteintion weights
        weights = tf.nn.softmax(logits, name=self.name + "_attention_weights", axis=-1)
        # weights = tf.nn.sigmoid(logits, name=self.name + "_attention_weights")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout_rate > 0:
            weights = self.dropout(weights)

        return tf.matmul(weights, v)

