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


def tokenize(string):
    token_list = string.strip().split()
    return token_list


def stringize(index_list, vocabulary):
    token_list = list()
    for index in index_list:
        if index == vocabulary.eos_index:
            break
        if index in {vocabulary.pad_index, vocabulary.bos_index}:
            continue
        else:
            token_list.append(vocabulary.token(index))
    return token_list


def numericalize(token_list, vocabulary):
    bos_index = vocabulary.bos_index
    eos_index = vocabulary.eos_index
    index_list = [ bos_index ] + [ vocabulary.index(token) for token in token_list ] + [ eos_index ]
    return index_list
