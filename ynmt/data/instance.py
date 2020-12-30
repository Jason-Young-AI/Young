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


class Instance(object):
    def __init__(self, **attributes):
        self.__structure = set(attributes.keys())
        for attribute_name, attribute_value in attributes.items():
            self[attribute_name] = attribute_value

    @property
    def structure(self):
        return self.__structure

    def __setitem__(self, attribute_name, attribute_value):
        assert attribute_name in self.__structure, f'Attribute name is not defined, only {self.structure} are allowed .'
        self.__dict__[attribute_name] = attribute_value

    def __getitem__(self, attribute_name):
        assert attribute_name in self.__structure, f'Attribute name is not defined, only {self.structure} are allowed .'
        return self.__dict__[attribute_name]

    def __iter__(self):
        for attribute_name in self.structure:
            yield (attribute_name, self[attribute_name])

    def __repr__(self):
        repres = str()
        repres += 'Instance('
        for index, (attribute_name, attribute_value) in enumerate(self):
            repres += f'{attribute_name}={attribute_value!r}'
            if index != len(self.structure) - 1:
                repres += ', '
        repres += ')'
        return repres

    def __eq__(self, other):
        flag = True
        if self.structure != other.structure:
            flag = False
        else:
            for attribute_name, attribute_value in self:
                if attribute_value != other[attribute_name]:
                    flag = False

        return flag
