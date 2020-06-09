#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-09 03:02
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def numericalize(string_line, vocabulary):
    bos_index = vocabulary.bos_index
    eos_index = vocabulary.eos_index
    token_list = string_line.strip().split()
    index_list = [ bos_index ] + [ vocabulary.index(token) for token in token_list ] + [ eos_index ]
    return index_list
