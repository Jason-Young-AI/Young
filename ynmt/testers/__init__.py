#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-08-12 18:31
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

from yoolkit.registration import Registration, import_modules

from ynmt.testers.tester import Tester


tester_registration = Registration(Tester)


def build_tester(args, task, device_descriptor, logger):
    return tester_registration[args.name].setup(args, task, device_descriptor, logger)


def register_tester(registration_name):
    return tester_registration.register(registration_name)


import_modules('ynmt.testers', os.path.dirname(__file__))
