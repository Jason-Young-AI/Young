#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-21 16:33
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


non_linear_dim = set({'Softmax', 'LogSoftmax', 'Softmin'})
non_linear_nodim = set({'ReLU', 'Tanh', 'Sigmoid'})
non_linear = set()
non_linear.update(non_linear_dim)
non_linear.update(non_linear_nodim)


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, dimensions=list(), activation='Softmax'):
        super(MultilayerPerceptron, self).__init__()
        assert isinstance(dimensions, list), f'Dimensions should be List()'
        assert activation in non_linear, 'Activation {activation} not supported yet'

        input_dimensions = [input_dimension] + dimensions
        output_dimensions = dimensions + [output_dimension]
        self.io_sizes = list(zip(input_dimensions, output_dimensions))

        self.linear_layers = torch.nn.ModuleList()
        for input_dimension, output_dimension in self.io_sizes:
            linear_layer = torch.nn.Linear(input_dimension, output_dimension)
            self.linear_layers.append(linear_layer)
        if activation in non_linear_dim:
            self.activation = getattr(torch.nn, activation)(dim=-1)
        else:
            self.activation = getattr(torch.nn, activation)()
            
    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = linear_layer(x)

        x = self.activation(x)
        return x
