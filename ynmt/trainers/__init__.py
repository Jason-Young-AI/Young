#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:09
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.trainers.trainer import Trainer


trainer_registration = Registration(Trainer)


def build_trainer(settings, factory, model, scheduler, optimizer, tester, device_descriptor, logger, visualizer):
    return trainer_registration[settings.name].setup(settings, factory, model, scheduler, optimizer, tester, device_descriptor, logger, visualizer)


def register_trainer(registration_name):
    return trainer_registration.register(registration_name)


import_modules(os.path.dirname(__file__), package='ynmt.trainers')
