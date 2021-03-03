#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-12-17 14:17
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from youngs.schedulers import register_scheduler, Scheduler


@register_scheduler('fixed')
class Fixed(Scheduler):
    def __init__(self, fixed_learning_rate):
        super(Fixed, self).__init__()
        self.fixed_learning_rate = fixed_learning_rate

    def learning_rate(self, step):
        return self.fixed_learning_rate

    def state_dict(self):
        state_dict = dict()
        state_dict['fixed_learning_rate'] = self.fixed_learning_rate
        return state_dict

    def load_state_dict(self, state_dict):
        self.fixed_learning_rate = state_dict['fixed_learning_rate']

    @classmethod
    def setup(cls, settings, model):
        args = settings.args
        scheduler = cls(args.fixed_learning_rate)
        return scheduler
