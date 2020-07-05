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


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dimension, head_number, dropout_probability)
        super(MultiHeadAttention, self).__init__()
        head_dimension, remaind_dimension = divmod(dimension,  head_number)
        assert remaind_dimension== 0, "An Error occured during initialization of MultiHeadAttention."
        self.dimension = dimension
        self.head_number = head_number
        self.head_dimension = head_dimension

        self.query_linear = torch.nn.Linear(self.dimension, self.head_number * self.head_dimension)
        self.key_linear = torch.nn.Linear(self.dimension, self.head_number * self.head_dimension)
        self.value_linear = torch.nn.Linear(self.dimension, self.head_number * self.head_dimension)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_probability)

        self.attention_linear = torch.nn.Linear(self.dimension, self.head_number * self.head_dimension)

    def forward(self, query, key, value, attention_weight_mask):
        query_length, batch_size, dimension = query

        def split(x):
            return x.reshape(-1, batch_size, self.head_number, self.head_dimension).transpose(0, 1)

        def merge(x):
            return x.transpose(0, 1).reshape(-1, batch_size, self.head_number * self.head_dimension)

        query = self.query_linear(query)
        key = self.value_linear(key)
        value = self.value_linear(value)

        query = split(query)
        key = split(key)
        value = split(value)

        attention_weight = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.head_dimension)
        attention_weight = self.dropout(self.softmax(attention_weight)))
        attention_weight = attention_weight.masked_fill(attention_weight_mask.unsqueeze(1), float("-inf"))
        attention = torch.matmul(attention_weight, value)

        attention = merge(attention)
        attention = self.attention_linear(attention)

        return attention, attention_weight
