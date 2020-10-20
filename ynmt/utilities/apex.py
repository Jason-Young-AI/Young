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


import importlib


def get_apex():
    result = importlib.util.find_spec("apex")

    if result is not None:
        apex = importlib.import_module(apex)
    else:
        apex = None

    return apex


def mix_precision(ynmt_model, ynmt_optimizer, mix_precision=False, optimization_level='O0'):
    apex = get_apex()
    if apex is not None and mix_precision:
        ynmt_model, ynmt_optimizer.optimizer = apex.amp.initialize(ynmt_model, ynmt_optimizer.optimizer, opt_level=optimization_level)
        ynmt_optimizer.mix_precision = mix_precision
    return ynmt_model, ynmt_optimizer


def backward(loss, optimizer, mix_precision=False):
    apex = get_apex()
    if apex is not None and mix_precision:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
