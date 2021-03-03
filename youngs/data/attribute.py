#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:04
#
# This source code is licensed under the Apache-2.0 license found in the
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
