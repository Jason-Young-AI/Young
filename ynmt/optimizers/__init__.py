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


from ynmt.optimizers.optimizer import Optimizer


from ynmt.optimizers.adam import build_optimizer_adam


def build_optimizer(args, model):
    return globals()[f'build_optimizer_{args.name}'](args, model)
