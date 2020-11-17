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


class Dataset(object):
    def __init__(self, structure):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), 'Type of {attribute_name} in structure should be str().'
        self.__structure = structure
        self.__instances = list()

    def __getitem__(self, index):
        return self.instances[index]

    def __contains__(self, attribute_name):
        return attribute_name in self.structure

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    @property
    def structure(self):
        return self.__structure

    @property
    def instances(self):
        return self.__instances

    def add(self, instance):
        if instance.structure != self.structure:
            raise ValueError('The structure of instance and dataset do not match!')
        self.__instances.append(instance)

    def remove(self, index):
        del self.__instances[index]
