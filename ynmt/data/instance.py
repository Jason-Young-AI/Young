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


from yoolkit.statistics import Statistics


class Instance(object):
    def __init__(self, structure):
        assert isinstance(structure, set), f'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), f'Type of {attribute_name} in structure should be str().'

        self.__structure = structure
        for attribute_name in self.__structure:
            setattr(self, attribute_name, None)

    def __len__(self):
        return len(self.__structure)

    def __setitem__(self, attribute_name, attribute_value):
        if attribute_name not in self.__structure:
            self.__structure.add(attribute_name)
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
