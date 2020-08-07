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


import torch


from ynmt.data.attribute import pad_attribute


def pad_batch(batch, vocabularies, attribute_names, device_descriptor):
    assert isinstance(attribute_names, set), f'#2 argument : {attribute_names} must be a Set()'

    for attribute_name in attribute_names:
        padded_attributes, _ = pad_attribute(batch[attribute_name], vocabularies[attribute_name].pad_index)
        batch[attribute_name] = torch.tensor(padded_attributes, dtype=torch.long, device=device_descriptor)

    return batch


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
