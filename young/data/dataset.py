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


class Dataset(object):
    def __init__(self, structure):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), f'Type of {attribute_name} in structure should be str().'
        self.__structure = structure
        self.__instances = list()

    @property
    def structure(self):
        return self.__structure

    @property
    def instances(self):
        return self.__instances

    def __len__(self):
        return len(self.__instances)

    def __iter__(self):
        for instance in self.__instances:
            yield instance

    def __repr__(self):
        return f'Dataset({self.structure!r}) # Maybe there are Instance(s) in Dataset!'

    def __getitem__(self, index):
        return self.__instances[index]

    def add(self, instance):
        assert isinstance(instance, Instance), f'Type of instance should be Instance().'
        if instance.structure != self.structure:
            raise ValueError(f'The structure of instance and dataset does not match: {instance.structure} and {self.structure}!')

        self.__instances.append(instance)

    def remove(self, index):
        assert index < len(self), f'Index out of range.'
        self.__instances.pop(index)
