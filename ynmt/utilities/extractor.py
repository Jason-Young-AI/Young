#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-28 18:02
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def get_position(tensor, dim):
    pass


def get_mask(attend_scope):
    position = torch.arange(0, attend_scope.size(1)).repeat(attend_scope.size(0), attend_scope.size(1), 1)
    mask = torch.ge(position, attend_scope.unsqueeze(-1))
