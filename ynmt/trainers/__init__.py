#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:56
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.trainers.trainer import Trainer


trainer_registration = Registration(Trainer)


def build_trainer(args, task, model, scheduler, optimizer, device_descriptor, logger, visualizer):
    return trainer_registration[args.name].setup(args, task, model, scheduler, optimizer, device_descriptor, logger, visualizer)


def register_trainer(registration_name):
    return trainer_registration.register(registration_name)


import_modules('ynmt.trainers', os.path.dirname(__file__))
