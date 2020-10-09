#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-09-07 16:52
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

from ynmt.tasks.task import Task

from ynmt.utilities.registration import Registration, import_modules


task_registration = Registration(Task)


def build_task(args, logger):
    return task_registration[args.name].setup(args, logger)


def register_task(registration_name):
    return task_registration.register(registration_name)


import_modules('ynmt.tasks', os.path.dirname(__file__))
