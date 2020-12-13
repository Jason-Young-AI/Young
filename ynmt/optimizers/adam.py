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


import torch

from ynmt.optimizers import register_optimizer, Optimizer


@register_optimizer('adam')
class Adam(Optimizer):
    def __init__(self, parameters, learning_rate=0.001, betas=(0.9, 0.999), epsilon=1e-08):
        adam_optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=epsilon
        )
        super(Adam, self).__init__(adam_optimizer)

    @classmethod
    def setup(cls, settings, model):
        args = settings.args
        parameters = (parameter for parameter in model.parameters() if parameter.requires_grad and parameter.is_leaf)
        optimizer = cls(
            parameters,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            epsilon=args.epsilon
        )

        return optimizer
