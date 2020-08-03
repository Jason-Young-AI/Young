#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-30 09:38
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ynmt.criterions.criterion import Criterion
from ynmt.criterions.cross_entropy import build_criterion_cross_entropy
from ynmt.criterions.label_smoothing_cross_entropy import build_criterion_label_smoothing_cross_entropy


def build_criterion(args, vocabulary, device_descriptor):
    criterion = globals()[f'build_criterion_{args.name}'](args, vocabulary)
    criterion.to(device_descriptor)
    return criterion
