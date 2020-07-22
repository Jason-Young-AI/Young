#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-05 20:20
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math


def prediction_accuracy(correct_prediction, total_prediction):
    return correct_prediction / total_prediction


def per_prediction_cross_entropy(prediction_cross_entropy, total_prediction):
    return prediction_cross_entropy / total_prediction


def perplexity(per_prediction_cross_entropy):
    per_prediction_cross_entropy = min(per_prediction_cross_entropy, 512)
    return math.pow(2, per_prediction_cross_entropy)


class Statistics(object):
    def __init__(self, structure):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), 'Type of {attribute_name} in structure should be str().'
        self.__structure = structure

        for attribute_name in self.structure:
            setattr(self, attribute_name, 0)

    def __setitem__(self, attribute_name, attribute_value):
        self.__dict__[attribute_name] = attribute_value

    def __getitem__(self, attribute_name):
        return self.__dict__[attribute_name]

    @property
    def structure(self):
        return self.__structure
