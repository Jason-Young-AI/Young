#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-05 00:37
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def shuffled(sequence):
    indices = range(len(iterable_object))
    random.shuffle(indices)
    shuffled_sequence = [ sequence[index] for index in indices ]
    return shuffled_sequence
