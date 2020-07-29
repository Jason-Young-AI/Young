#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-26 09:44
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, dimension, overparameterized_dimension, dropout_probability):
        super(PositionWiseFeedForward, self).__init__()
        self.dimension = dimension

        self.dropout = torch.nn.Dropout(dropout_probability)

        self.full_connected_input_to_hidden = torch.nn.Linear(dimension, overparameterized_dimension)
        self.relu = torch.nn.ReLU(inplace=True)
        self.full_connected_hidden_to_output = torch.nn.Linear(overparameterized_dimension, dimension)

    def forward(self, x):
        x = self.full_connected_input_to_hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.full_connected_hidden_to_output(x)
        return x
