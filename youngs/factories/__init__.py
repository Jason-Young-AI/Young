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

from youngs.factories.factory import Factory


factory_registration = Registration(Factory)


def build_factory(settings, logger):
    return factory_registration[settings.name].setup(settings, logger)


def register_factory(registration_name):
    return factory_registration.register(registration_name)


import_modules(os.path.dirname(__file__), package='youngs.factories')
