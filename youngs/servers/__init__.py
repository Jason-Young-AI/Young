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

from youngs.servers.server import Server


server_registration = Registration(Server)


def build_server(settings, tester, logger):
    return server_registration[settings.name].setup(settings, tester, logger)


def register_server(registration_name):
    return server_registration.register(registration_name)


import_modules(os.path.dirname(__file__), 'youngs.servers')
