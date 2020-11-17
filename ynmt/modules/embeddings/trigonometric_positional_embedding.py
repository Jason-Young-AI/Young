#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:06
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch


class TrigonometricPositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_number, dimension, padding_idx=None):
        super(TrigonometricPositionalEmbedding, self).__init__()
        self.embedding_number = embedding_number
        self.dimension = dimension
        self.padding_idx = padding_idx
        position = torch.arange(0, self.embedding_number).unsqueeze(1)
        sin_multiplicator = torch.exp(-(math.log(10000) / self.dimension) * torch.arange(0, self.dimension, 2))
        cos_multiplicator = torch.exp(-(math.log(10000) / self.dimension) * torch.arange(1, self.dimension, 2))
        sin_weight = torch.sin(position * sin_multiplicator)
        cos_weight = torch.cos(position * cos_multiplicator)

        weight = torch.zeros(self.embedding_number, self.dimension)
        weight[:, 0::2] = sin_weight
        weight[:, 1::2] = cos_weight
        weight[self.padding_idx] = 0

        self.register_buffer('weight', weight)

    def forward(self, x):
        # x: [* x 1] or [* x Dimension]
        x = x + self.weight[0:x.size(-2)]
        return x
