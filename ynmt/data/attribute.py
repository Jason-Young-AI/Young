#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-20 18:15
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def pad_attribute(attributes, pad_index):
    max_attribute_length = max([len(attribute) for attribute in attributes])
    padded_attributes = list()
    attribute_lengths = list()
    for attribute in attributes:
        attribute_length = len(attribute)
        attribute_lengths.append(attribute_length)
        padded_attribute = attribute + [pad_index] * (max_attribute_length - attribute_length)
        padded_attributes.append(padded_attribute)

    return padded_attributes, attribute_lengths
