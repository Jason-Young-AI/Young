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


non_linear_dim = set({'Softmax', 'LogSoftmax', 'Softmin'})
non_linear_nodim = set({'ReLU', 'Tanh', 'Sigmoid'})


class MultilayerPerceptron(torch.nn.Module):
    def __init__(
        self, input_dimension, output_dimension, bias_flag,
        hidden_layer_dimensions=list(), hidden_layer_bias_flags=list(), activation=None
    ):
        super(MultilayerPerceptron, self).__init__()
        assert isinstance(hidden_layer_dimensions, list), f'Dimensions of hidden layer should be List()'
        assert isinstance(hidden_layer_bias_flags, list), f'Bias flags of hidden layer should be List()'
        assert len(hidden_layer_dimensions) == len(hidden_layer_bias_flags), f'Length of hidden_layer_dimensions & hidden_layer_dimensions mismatch'

        input_dimensions = [input_dimension] + hidden_layer_dimensions
        output_dimensions = hidden_layer_dimensions + [output_dimension]
        self.io_sizes = list(zip(input_dimensions, output_dimensions))

        self.bias_flags = [bias_flag] + hidden_layer_bias_flags

        self.linear_layers = torch.nn.ModuleList()
        for (input_dimension, output_dimension), bias_flag in zip(self.io_sizes, self.bias_flags):
            linear_layer = torch.nn.Linear(input_dimension, output_dimension, bias=bias_flag)
            self.linear_layers.append(linear_layer)

        if activation in non_linear_dim:
            self.activation = getattr(torch.nn, activation)(dim=-1)
        elif activation in non_linear_nodim:
            if activation == 'ReLU':
                self.activation = getattr(torch.nn, activation)(inplace=True)
            else:
                self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = lambda x: x
            
    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        x = self.activation(x)
        return x
