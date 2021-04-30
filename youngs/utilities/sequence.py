#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:10
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import re


def tokenize(string):
    token_list = string.strip().split()
    return token_list


def stringize(index_list, vocabulary):
    token_list = list()
    for index in index_list:
        if index == vocabulary.eos_index:
            break
        if index in set(vocabulary.reserved_tokens):
            continue
        else:
            token_list.append(vocabulary.token(index))
    return token_list


def numericalize(token_list, vocabulary, add_bos=True, add_eos=True):
    index_list = [ vocabulary.index(token) for token in token_list ]

    if add_bos:
        index_list = [ vocabulary.bos_index ] + index_list

    if add_eos:
        index_list = index_list + [ vocabulary.eos_index ]

    return index_list
