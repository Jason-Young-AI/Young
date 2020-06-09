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

    def __iter__(self):
        for attribute_name in self.structure:
            yield self.__dict__[attribute_name]

    @property
    def structure(self):
        return self.__structure
