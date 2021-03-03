#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-12-10 08:41
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

from youngs.optimizers import register_optimizer, Optimizer


@register_optimizer('sgd')
class SGD(Optimizer):
    def __init__(self, parameters, learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        sgd_optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super(SGD, self).__init__(sgd_optimizer)

    @classmethod
    def setup(cls, settings, model):
        args = settings.args
        parameters = (parameter for parameter in model.parameters() if parameter.requires_grad and parameter.is_leaf)
        optimizer= cls(
            parameters,
            args.learning_rate,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )

        return optimizer
