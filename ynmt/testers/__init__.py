#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:08
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.testers.tester import Tester


tester_registration = Registration(Tester)


def build_tester(settings, factory, model, device_descriptor, logger):
    return tester_registration[settings.name].setup(settings, factory, model, device_descriptor, logger)


def register_tester(registration_name):
    return tester_registration.register(registration_name)


import_modules(os.path.dirname(__file__), package='ynmt.testers')
