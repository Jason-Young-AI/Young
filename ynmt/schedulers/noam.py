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


from ynmt.schedulers import register_scheduler, Scheduler


@register_scheduler('noam')
class Noam(Scheduler):
    def __init__(self, dimension, warmup_step):
        self.dimension = dimension
        self.warmup_step = warmup_step
        super(Noam, self).__init__()

    def learning_rate(self, step):
        learning_rate = min(step ** (-0.5), step * self.warmup_step ** (-1.5)) * (self.dimension ** (-0.5))
        return learning_rate

    def state_dict(self):
        state_dict = dict()
        state_dict['warmup_step'] = self.warmup_step
        state_dict['dimension'] = self.dimension
        return state_dict

    def load_state_dict(self, state_dict):
        self.warmup_step = state_dict['warmup_step']
        self.dimension = state_dict['dimension']

    @classmethod
    def setup(cls, settings, model):
        args = settings.args
        noam = cls(model.dimension, args.warmup_step)
        return noam
