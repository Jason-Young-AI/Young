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


from ynmt.data.instance import Instance


class Batch(object):
    def __init__(self, structure):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), f'Type of {attribute_name} in structure should be str().'
        self.__structure = structure
        self.__size = 0
        for attribute_name in self.structure:
            self[attribute_name] = list()

    @property
    def structure(self):
        return self.__structure

    def __len__(self):
        return self.__size

    def __iter__(self):
        for attribute_name in self.structure:
            yield (attribute_name, self[attribute_name])

    def __repr__(self):
        return f'Batch({self.structure!r}) # Maybe there are Instance(s) in Batch!'

    def __setitem__(self, attribute_name, attribute_value):
        assert attribute_name in self.__structure, f'Attribute name is not defined, only {self.structure} are allowed .'
        self.__dict__[attribute_name] = attribute_value

    def __getitem__(self, attribute_name):
        assert attribute_name in self.__structure, f'Attribute name is not defined, only {self.structure} are allowed .'
        return self.__dict__[attribute_name]

    def add(self, instance):
        assert isinstance(instance, Instance), f'Type of instance should be Instance().'
        if instance.structure != self.structure:
            raise ValueError(f'The structure of instance and batch does not match: {instance.structure} and {self.structure}!')

        self.__size += 1
        for attribute_name, attribute_value in instance:
            self[attribute_name].append(attribute_value)

    def remove(self, index):
        assert index < len(self), f'Index out of range.'

        self.__size -= 1
        for attribute_name in self.structure:
            self[attribute_name].pop(index)
