#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-23 17:09
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.optimizers.optimizer import Optimizer


optimizer_registration = Registration(Optimizer)


def build_optimizer(settings, model):
    return optimizer_registration[settings.name].setup(settings, model)


def register_optimizer(registration_name):
    return optimizer_registration.register(registration_name)


import_modules('ynmt.optimizers', os.path.dirname(__file__))
