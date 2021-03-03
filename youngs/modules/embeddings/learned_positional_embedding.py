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


import torch


class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_number, dimension, padding_idx=None):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(embedding_number, dimension))

    def forward(self, x, index=None):
        # x: [* x 1] or [* x Dimension]
        if index is None:
            x = x + self.weight[0:x.size(-2)]
        else:
            x = x + self.weight[index]
        return x
