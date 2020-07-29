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


from ynmt.utilities.statistics import Statistics


class InstanceFilter(object):
    def __init__(self, length_intervals):
        self.length_intervals = length_intervals

    def __call__(self, instance):
        for attribute_name, length_interval in self.length_intervals.items():
            assert attribute_name in instance.structure, f"No such attribute:{attribute_name} in instance:{instance}"
            attribute_length = len(instance[attribute_name])
            if attribute_length < length_interval[0] or length_interval[1] < attribute_length:
                return True
        return False


class InstanceSizeCalculator(object):
    def __init__(self, calculate_type):
        assert calculate_type in {'token', 'sentence'}, "Wrong choice of calculator type."

        self.calculate_type = calculate_type

    def __call__(self, instance):
        attribute_sizes = Statistics(set())
        for attribute_name in instance.structure:
            if self.calculate_type == 'sentence':
                attribute_size = 1
            if self.calculate_type == 'token':
                attribute_size = len(instance[attribute_name])

            attribute_sizes[attribute_name] = attribute_size

        return attribute_sizes


class InstanceComparator(object):
    def __init__(self, comparision_attrs=[]):
        assert isinstance(comparision_attrs, list), "#1 arg {comparision_attributes} should be a List()."
        self.comparision_attrs = set()
        self.comparision_order = list()

        for comparision_attr in comparision_attrs:
            if comparision_attr in self.comparision_attrs:
                continue
            else:
                self.comparision_attrs.add(comparision_attr)
                self.comparision_order.append(comparision_attr)

    def __call__(self, instance):
        self.optional_attrs = instance.structure - self.comparision_attrs
        self.required_attrs = instance.structure - self.optional_attrs

        comparision_list = list()
        for comparision_attr in self.comparision_order:
            if comparision_attr in self.required_attrs:
                comparision_list.append(len(instance[comparision_attr]))

        for optional_attr in self.optional_attrs:
            comparision_list.append(len(instance[optional_attr]))

        return tuple(comparision_list)


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

    def __contains__(self, attribute_name):
        return attribute_name in self.structure

    def __iter__(self):
        for attribute_name in self.structure:
            yield (attribute_name, self[attribute_name])

    @property
    def structure(self):
        return self.__structure
