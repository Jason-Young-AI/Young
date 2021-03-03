#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch


class Model(torch.nn.Module):
    def __init__(self, settings):
        super(Model, self).__init__()
        self.settings = settings

    @classmethod
    def setup(cls, settings, factory):
        raise NotImplementedError

    def personalized_load_state(self, model_state):
        raise NotImplementedError
