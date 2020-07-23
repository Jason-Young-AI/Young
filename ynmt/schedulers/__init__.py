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


from ynmt.schedulers.scheduler import Scheduler
from ynmt.schedulers.noam import build_scheduler_noam


def build_scheduler(args, model, checkpoint):
    scheduler = globals()[f'build_scheduler_{args.name}'](args, model)

    if checkpoint is not None and not args.reset_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return scheduler
