#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch


from ynmt.modules.embeddings import TrigonometricPositionalEmbedding
from ynmt.modules.attentions import MultiHeadAttention
from ynmt.modules.perceptrons import PositionWiseFeedForward


class TransformerDecoder(torch.nn.Module):
    def __init__(self, vocabulary, 
                 layer_number, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerDecoder, self).__init__()
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.embed_token = torch.nn.Embedding(len(vocabulary), dimension, padding_idx=vocabulary.pad_index)
        self.embed_position = TrigonometricPositionalEmbedding(5000, dimension, padding_idx=vocabulary.pad_index)

        self.transformer_decoder_layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dimension, feedforward_dimension, head_number,
                    dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                    normalize_position
                )
                for _ in range(layer_number)
            ]
        )
        self.final_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

    def embed(self, x):
        x = self.embed_token(x)
        x = x * math.sqrt(self.dimension)
        x = self.embed_position(x)
        x = self.dropout(x)
        return x

    def forward(self, target, codes, self_attention_weight_mask, cross_attention_weight_mask):
        # target: [Batch_Size x Target_Length],
        # codes: [Batch_Size x Target_Length],
        # self_attention_weight_mask: [Batch_Size x Source_Length x Source_Length]
        # cross_attention_weight_mask: [Batch_Size x Target_Length x Source_Length]
        x = self.embed(target)

        for index, transformer_decoder_layer in enumerate(self.transformer_decoder_layers):
            x, cross_attention_weight  = transformer_decoder_layer(x, codes, self_attention_weight_mask, cross_attention_weight_mask)

        x = self.final_normalization(x)

        return x, cross_attention_weight


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerDecoderLayer, self).__init__()
        assert normalize_position in {'before', 'after'}, 'Only support \'before\' and \'after\' normalize position'
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.normalize_position = normalize_position

        self.self_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.self_attention_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

        self.cross_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.cross_attention_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

        self.positionwise_feedforward = PositionWiseFeedForward(dimension, feedforward_dimension, feedforward_dropout_probability)
        self.positionwise_feedforward_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

    def forward(self, x, codes, self_attention_weight_mask, cross_attention_weight_mask):
        # self attention sublayer
        residual = x
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'before')
        x, _ = self.self_attention(query=x, key=x, value=x, attention_weight_mask=self_attention_weight_mask)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'after')

        # cross attention sublayer
        residual = x
        x = layer_normalize(x, self.cross_attention_normalization, self.normalize_position == 'before')
        x, cross_attention_weight = self.cross_attention(query=x, key=codes, value=codes, attention_weight_mask=cross_attention_weight_mask)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.cross_attention_normalization, self.normalize_position == 'after')

        # position-wise feed-forward sublayer
        residual = x
        x = layer_normalize(x, self.positionwise_feedforward_normalization, self.normalize_position == 'before')
        x = self.positionwise_feedforward(x)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.positionwise_feedforward_normalization, self.normalize_position == 'after')

        return x, cross_attention_weight


def layer_normalize(x, normalization, do):
    if do:
        return normalization(x)
    else:
        return x
