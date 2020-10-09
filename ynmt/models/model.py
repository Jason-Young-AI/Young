#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-08-11 17:12
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args 

    @classmethod
    def setup(cls, args, task):
        raise NotImplementedError
