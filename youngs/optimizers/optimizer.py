#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from youngs.utilities.apex import backward


class Optimizer(object):
    def __init__(self, optimizer, mix_precision=False):
        self.optimizer = optimizer
        self.mix_precision = mix_precision

    @classmethod
    def setup(cls, settings, model):
        raise NotImplementedError

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

    def backward(self, loss):
        backward(loss, self.optimizer, self.mix_precision)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
