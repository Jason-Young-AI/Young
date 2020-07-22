#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-23 17:09
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


class Optimizer(object):
    def __init__(self, optimizer):
        self.current_step = 1
        self.optimizer = optimizer

    @property
    def defaults(self):
        return self.optimizer.defaults

    @property
    def parameter_groups(self):
        return self.optimizer.param_groups

    @property
    def parameters(self):
        for parameter_group in self.parameter_groups:
            parameters = parameter_group['params']
            for parameter in parameters:
                yield parameter

    @property
    def learning_rates(self):
        learning_rates = dict()
        for index, parameter_group in enumerate(self.parameter_groups):
            learning_rates[f'parameter_group-{index}'] = parameter_group['lr']
        return learning_rates

    def step(self):
        self.optimizer.step()
        self.current_step += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dict = dict()
        state_dict['current_step'] = self.current_step
        state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.optimizer.load_state_dict(state_dict['optimizer'])
