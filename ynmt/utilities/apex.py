#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-10-14 10:44
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import apex


def mix_precision(ynmt_model, ynmt_optimizer, mix_precision=False, optimization_level='O0'):
    if mix_precision:
        ynmt_model, ynmt_optimizer.optimizer = apex.amp.initialize(ynmt_model, ynmt_optimizer.optimizer, opt_level=optimization_level)
        ynmt_optimizer.mix_precision = mix_precision
    return ynmt_model, ynmt_optimizer
