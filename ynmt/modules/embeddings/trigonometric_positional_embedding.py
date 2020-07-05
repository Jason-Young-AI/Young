#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-27 07:47
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch


class TrigonometricPositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_number, dimension):
        position = torch.arange(0, embedding_number).unsqueeze(1)
        sin_multiplicator = torch.exp(-(math.log(10000) / dimension) * 2 * torch.arange(0, dimension, 2))
        cos_multiplicator = torch.exp(-(math.log(10000) / dimension) * 2 * torch.arange(1, dimension, 2))
        sin_weight = torch.sin(position * sin_multiplicator)
        cos_weight = torch.cos(position * cos_multiplicator)

        weight = torch.zeros(embedding_number, dimension)
        weight[:, 0::2] = sin_weight
        weight[:, 1::2] = cos_weight
        weight[self.padding_idx] = 0
        self.register_buffer('weight', weight)

    def forward(self, position):
        torch.index_select(self.weight, 0, position)
