#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-29 21:56
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ynmt.scheduler import Scheduler


def build_scheduler_noam(args, model, checkpoint):
    noam = Noam(model.dimension, args.warmup_step)
    return noam


class Noam(Scheduler):
    def __init__(self, dimension, warmup_step):
        self.dimension = dimension
        self.warmup_step = warmup_step

    def learning_rate(self, step):
        learning_rate = min(step ** (-0.5), step * self.warmup_step ** (-1.5)) * (self.dimension ** (-0.5))
        return learning_rate
