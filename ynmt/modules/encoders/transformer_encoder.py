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


import torch


from ynmt.modules.embeddings import TrigonometricPositionalEmbedding
from ynmt.modules.attentions import MultiHeadAttention
from ynmt.modules.perceptrons import PositionWiseFeedForward
from ynmt.utilities.extractor import get_position, get_attend_mask


class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocabulary, 
                 layer_number, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerEncoder, self).__init__()
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.embed_token = torch.nn.Embedding(len(vocabulary), dimension, padding_idx=vocabulary.pad_index)
        self.embed_position = TrigonometricPositionalEmbedding(2048, dimension, padding_idx=vocabulary.pad_index)
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

    def embed(self, source):
        token = source
        position = get_position(source.transpose(0, 1)).transpose(0, 1)
        token_embedding = self.embed_token(token)
        position_embedding = self.embed_position(position)
        embedding = token_embedding + position_embedding
        embedding = self.dropout(embedding)
        return embedding

    def forward(self, source, source_length):
        # calculate attention weight mask
        x = self.embed(source)

        source_max_length = torch.max(source_length)
        source_attend_scope = source_length.unsqueeze(1).repeat(1, source_max_length)
        attention_weight_mask = get_attend_mask(source_max_length, source_attend_scope)

        for index, transformer_encoder_layer in enumerate(self.transformer_encoder_layers):
            x, attention_weight = transformer_encoder_layer(x, attention_weight_mask)

        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerEncoderLayer, self).__init__()
        assert normalize_position in {'before', 'after'}, 'Only support \'before\' and \'after\' normalize position'
        self.normalize_position = normalize_position

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.self_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.self_attention_normalization = torch.nn.LayerNorm(dimension)
        self.positionwise_feedforward = PositionWiseFeedForward(dimension, feedforward_dimension, feedforward_dropout_probability)
        self.positionwise_feedforward_normalization = torch.nn.LayerNorm(dimension)


    def forward(self, x, x_mask):
        # self attention sublayer
        residual = x
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'before')
        x, attention_weight = self.self_attention(query=x, key=x, value=x, attention_weight_mask=x_mask)
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

        return x, attention_weight



def layer_normalize(x, normalization, do):
    if do:
        return normalization(x)
    else:
        return x
