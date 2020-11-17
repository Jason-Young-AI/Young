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


import os

from yoolkit.registration import Registration, import_modules

from ynmt.schedulers.scheduler import Scheduler


scheduler_registration = Registration(Scheduler)


def build_scheduler(settings, model):
    return scheduler_registration[settings.name].setup(settings, model)


def register_scheduler(registration_name):
    return scheduler_registration.register(registration_name)


import_modules('ynmt.schedulers', os.path.dirname(__file__))
