#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-05 18:36
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ynmt.testers.beam_search import build_tester_beam_search


def build_tester(args, vocabulary):
    return globals()[f'build_tester_{args.name}'](args, vocabulary)
