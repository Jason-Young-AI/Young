#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:04
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch

from ynmt.modules.embeddings import TrigonometricPositionalEmbedding
from ynmt.modules.attentions import MultiHeadAttention
from ynmt.modules.perceptrons import PositionWiseFeedForward


class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocabulary, 
                 layer_number, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerEncoder, self).__init__()
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.embed_token = torch.nn.Embedding(len(vocabulary), dimension, padding_idx=vocabulary.pad_index)
        self.embed_position = TrigonometricPositionalEmbedding(5000, dimension, padding_idx=vocabulary.pad_index)

        self.transformer_encoder_layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dimension, feedforward_dimension, head_number,
                    dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                    normalize_position
                )
                for _ in range(layer_number)
            ]
        )
        self.final_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

        self.initialize()

    def embed(self, x):
        x = self.embed_token(x)
        x = x * math.sqrt(self.dimension)
        x = self.embed_position(x)
        x = self.dropout(x)
        return x

    def forward(self, source, attention_weight_mask):
        # source: [Batch_Size x Source_Length],
        # attention_weight_mask: [Batch_Size x Source_Length x Source_Length]

        x = self.embed(source) # [Batch_Size x Source_Length x Dimension]

        for index, transformer_encoder_layer in enumerate(self.transformer_encoder_layers):
            x = transformer_encoder_layer(x, attention_weight_mask)

        x = self.final_normalization(x)
        return x

    def initialize(self):
        torch.nn.init.normal_(self.embed_token.weight, mean=0, std=self.embed_token.embedding_dim ** -0.5)
        torch.nn.init.constant_(self.embed_token.weight[self.embed_token.padding_idx], 0.0)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerEncoderLayer, self).__init__()
        assert normalize_position in {'before', 'after'}, 'Only support \'before\' and \'after\' normalize position'
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.normalize_position = normalize_position

        self.self_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.self_attention_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

        self.positionwise_feedforward = PositionWiseFeedForward(dimension, feedforward_dimension, feedforward_dropout_probability)
        self.positionwise_feedforward_normalization = torch.nn.LayerNorm(dimension, eps=1e-6)

    def forward(self, x, attention_weight_mask):
        # self attention sublayer
        residual = x
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'before')
        x, _ = self.self_attention(query=x, key=x, value=x, attention_weight_mask=attention_weight_mask)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'after')

        # position-wise feed-forward sublayer
        residual = x
        x = layer_normalize(x, self.positionwise_feedforward_normalization, self.normalize_position == 'before')
        x = self.positionwise_feedforward(x)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.positionwise_feedforward_normalization, self.normalize_position == 'after')

        return x


def layer_normalize(x, normalization, do):
    if do:
        return normalization(x)
    else:
        return x
