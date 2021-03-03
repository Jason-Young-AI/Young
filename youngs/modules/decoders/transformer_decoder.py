#!/usr/bin/env python3 -u
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

from youngs.modules.embeddings import TrigonometricPositionalEmbedding
from youngs.modules.attentions import MultiHeadAttention
from youngs.modules.perceptrons import PositionWiseFeedForward


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

        self.clear_caches()

        self.initialize()

    def embed(self, x, using_cache):
        if using_cache:
            index = self.caches['step']
        else:
            index = None

        x = self.embed_token(x)
        x = x * math.sqrt(self.dimension)
        x = self.embed_position(x, index)
        x = self.dropout(x)
        return x

    def forward(self, target, codes, self_attention_weight_mask, cross_attention_weight_mask, using_step_cache=False, using_self_cache=False, using_cross_cache=False):
        # target: [Batch_Size x Target_Length],
        # codes: [Batch_Size x Source_Length x Dimension],
        # self_attention_weight_mask: [Batch_Size x Target_Length x Target_Length],
        # cross_attention_weight_mask: [Batch_Size x Target_Length x Source_Length]

        using_cache = using_step_cache or using_self_cache or using_cross_cache

        if using_cache:
            self.initialize_caches()

        x = self.embed(target, using_cache=using_step_cache)

        for index, transformer_decoder_layer in enumerate(self.transformer_decoder_layers):
            if using_cache:
                attention_cache = self.caches['attention'][f'layer_{index}']
            else:
                attention_cache = None

            x, cross_attention_weight  = transformer_decoder_layer(x, codes, self_attention_weight_mask, cross_attention_weight_mask, cache=attention_cache, using_self_cache=using_self_cache, using_cross_cache=using_cross_cache)

        x = self.final_normalization(x)

        return x, cross_attention_weight

    def initialize(self):
        torch.nn.init.normal_(self.embed_token.weight, mean=0, std=self.embed_token.embedding_dim ** -0.5)
        torch.nn.init.constant_(self.embed_token.weight[self.embed_token.padding_idx], 0.0)

    def clear_caches(self):
        self.caches = None

    def initialize_caches(self):
        if self.caches is None:
            self.caches = dict(
                step = 0,
                attention = dict(),
            )

            for index, transformer_decoder_layer in enumerate(self.transformer_decoder_layers):
                attention_cache = dict(
                    self_keys = None,
                    self_values = None,
                    cross_keys = None,
                    cross_values = None,
                )

                self.caches['attention'][f'layer_{index}'] = attention_cache

    def update_caches(self, order):

        def recursive_reorder(cache):
            for key, value in cache.items():
                if value is not None:
                    if isinstance(value, dict):
                        recursive_reorder(value)
                    else:
                        cache[key] = value.index_select(0, order)

        if self.caches is not None:
            self.caches['step'] += 1
            recursive_reorder(self.caches['attention'])


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

    def forward(self, x, codes, self_attention_weight_mask, cross_attention_weight_mask, cache=None, using_self_cache=False, using_cross_cache=False):
        # self attention sublayer
        residual = x
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'before')
        x, _ = self.self_attention(
            query=x, key=x, value=x, attention_weight_mask=self_attention_weight_mask,
            attention_type='self', cache=cache if using_self_cache else None
        )
        x = self.dropout(x)
        x = x + residual
        x = layer_normalize(x, self.self_attention_normalization, self.normalize_position == 'after')

        # cross attention sublayer
        residual = x
        x = layer_normalize(x, self.cross_attention_normalization, self.normalize_position == 'before')
        x, cross_attention_weight = self.cross_attention(
            query=x, key=codes, value=codes, attention_weight_mask=cross_attention_weight_mask,
            attention_type='cross', cache=cache if using_cross_cache else None
        )
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
