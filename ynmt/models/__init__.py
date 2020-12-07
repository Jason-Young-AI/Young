#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.models.model import Model


model_registration = Registration(Model)


def build_model(settings, factory):
    return model_registration[settings.name].setup(settings, factory)


def register_model(registration_name):
    return model_registration.register(registration_name)


import_modules(os.path.dirname(__file__), package='ynmt.models')
