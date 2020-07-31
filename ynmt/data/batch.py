#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-03 14:21
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def pack_batch(batch_iterator, pack_size):
    packed_batch = list()

    for batch in batch_iterator:
        packed_batch.append(batch)
        if len(packed_batch) == pack_size:
            yield packed_batch
            packed_batch = list()

    if len(packed_batch) != 0:
        yield packed_batch


class Batch(object):
    def __init__(self, structure, instances=list()):
        assert isinstance(structure, set), 'Type of structure should be set().'
        for attribute_name in structure:
            assert isinstance(attribute_name, str), 'Type of {attribute_name} in structure should be str().'
        self.__structure = structure

        for attribute_name in self.structure:
            setattr(self, attribute_name, list())

        for instance in instances:
            for attribute_name in instance.structure:
                if attribute_name in self.structure:
                    attribute_value = getattr(self, attribute_name)
                    attribute_value.append(instance[attribute_name])

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
