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

    def __len__(self):
        return len(self.__structure)

    def __setitem__(self, attribute_name, attribute_value):
        self.__dict__[attribute_name] = attribute_value

    def __getitem__(self, attribute_name):
        if attribute_name in self.__structure:
            return self.__dict__[attribute_name]
        else:
            return 0

    def __contains__(self, attribute_name):
        return attribute_name in self.structure

    def __iter__(self):
        for attribute_name in self.structure:
            yield (attribute_name, self[attribute_name])

    def __str__(self):
        return self.__repr__()

    def __add__(self, other_statistics):
        result_structure = self.structure | other_statistics.structure
        result_statistics = Statistics(result_structure)
        for attribute_name in result_statistics.structure:
            result_statistics[attribute_name] = self[attribute_name] + other_statistics[attribute_name]
        return result_statistics

    def clear(self):
        for attribute_name in self.structure:
            self[attribute_name] = 0

    @property
    def structure(self):
        return self.__structure
