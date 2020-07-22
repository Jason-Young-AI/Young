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


import torch


from ynmt.modules.embeddings import TrigonometricPositionalEmbedding
from ynmt.modules.attentions import MultiHeadAttention
from ynmt.modules.perceptrons import PositionWiseFeedForward
from ynmt.utilities.extractor import get_position, get_attend_mask


class TransformerDecoder(torch.nn.Module):
    def __init__(self, vocabulary, 
                 layer_number, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerDecoder, self).__init__()
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.embed_token = torch.nn.Embedding(len(vocabulary), dimension, padding_idx=vocabulary.pad_index)
        self.embed_position = TrigonometricPositionalEmbedding(2048, dimension)
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

    def embed(self, target):
        token = target
        position = get_position(target.transpose(0, 1)).transpose(0, 1)
        token_embedding = self.embed_token(token)
        position_embedding = self.embed_position(position)
        embedding = token_embedding + position_embedding
        embedding = self.dropout(embedding)
        return embedding

    def forward(self, codes, codes_length, target, target_length):
        x = self.embed(target)

        # calculate self & cross attention weight mask
        target_max_length = torch.max(target_length)
        target_attend_scope = target_length.unsqueeze(1).repeat(1, target_max_length)
        target_attention_weight_mask = get_attend_mask(target_max_length, target_attend_scope)

        codes_max_length = torch.max(codes_length)
        codes_attend_scope = codes_length.unsqueeze(1).repeat(1, codes_max_length)
        codes_attention_weight_mask = get_attend_mask(target_max_length, codes_attend_scope)

        # calculate future mask
        future_attend_scope = torch.arange(1, target_max_length + 1, device=target_length.device).repeat(len(target_length), 1)
        future_attention_weight_mask = get_attend_mask(target_max_length, future_attend_scope)

        attention_weight_mask = future_attention_weight_mask | target_attention_weight_mask

        for transformer_decoder_layer in self.transformer_decoder_layers:
            x, self_attention_weight, cross_attention_weight  = transformer_decoder_layer(x, attention_weight_mask, codes, codes_attention_weight_mask)

        return x, self_attention_weight, cross_attention_weight


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, dimension, feedforward_dimension, head_number,
                 dropout_probability, attention_dropout_probability, feedforward_dropout_probability,
                 normalize_position):
        super(TransformerDecoderLayer, self).__init__()
        assert normalize_position in {'before', 'after'}, 'Only support \'before\' and \'after\' normalize position'
        self.normalize_position = normalize_position

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.self_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.self_attention_normalization = torch.nn.LayerNorm(dimension)
        self.cross_attention = MultiHeadAttention(dimension, head_number, attention_dropout_probability)
        self.cross_attention_normalization = torch.nn.LayerNorm(dimension)
        self.positionwise_feedforward = PositionWiseFeedForward(dimension, feedforward_dimension, feedforward_dropout_probability)
        self.positionwise_feedforward_normalization = torch.nn.LayerNorm(dimension)


    def forward(self, x, x_mask, codes, codes_mask):
        # self attention sublayer
        residual = x
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'before')
        x, self_attention_weight = self.self_attention(query=x, key=x, value=x, attention_weight_mask=x_mask)
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'after')

        # cross attention sublayer
        residual = x
        x = layer_normalize(x, self.cross_attention_normalization, self.normalize_position == 'before')
        x, cross_attention_weight = self.cross_attention(query=x, key=codes, value=codes, attention_weight_mask=codes_mask)
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

        return x, self_attention_weight, cross_attention_weight


def layer_normalize(x, normalization, do):
    if do:
        return normalization(x)
    else:
        return x
