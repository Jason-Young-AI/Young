#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-06 02:09
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class InstanceFilter(object):
    def __init__(self, length_intervals):
        self.length_intervals = length_intervals

    def __call__(self, instance):
        for attribute_name, length_interval in self.length_intervals.items():
            assert attribute_name in instance.structure, f"No such attribute:{attribute_name} in instance:{instance}"
            attribute_length = len(instance[attribute_name])
            if length_interval[0] < attribute_length and attribute_length < length_interval[1]:
                return False
            else:
                return True


class InstanceSizeCalculator(object):
    def __init__(self, calculate_attribute, calculate_type):
        assert calculate_type in {'token', 'sentence'}, "Wrong choice of calculator type."

        self.calculate_attribute = calculate_attribute
        self.calculate_type = calculate_type

    def __call__(self, instance):
        if self.calculate_type == 'sentence':
            return 1

        if self.calculate_type == 'token':
            attribute = instance[self.calculate_attribute]
            return len(attribute)


class Instance(object):
    def __init__(self, structure):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), 'Type of {attribute_name} in structure should be str().'

        self.__structure = structure
        for attribute_name in self.__structure:
            setattr(self, attribute_name, None)

    def __len__(self):
        return len(self.__structure)

    def __setitem__(self, attribute_name, attribute_value):
        self.__dict__[attribute_name] = attribute_value

    def __getitem__(self, attribute_name):
        return self.__dict__[attribute_name]

    def __iter__(self):
        for attribute_name in self.structure:
            yield self.__dict__[attribute_name]

    @property
    def structure(self):
        return self.__structure
