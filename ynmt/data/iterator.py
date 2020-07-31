#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-03 14:05
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random


from ynmt.data.batch import Batch
from ynmt.data.instance import InstanceComparator
from ynmt.utilities.file import load_data_objects
from ynmt.utilities.random import shuffled
from ynmt.utilities.statistics import Statistics


class Iterator(object):
    def __init__(self, dataset_path, batch_size, instance_size_calculator, instance_filter=None, instance_comparator=InstanceComparator(), traverse_time=1, accumulate_number=1, mode='preserve'):
        assert mode in {'preserve', 'ascend', 'descend', 'shuffle'}, "Wrong choice of order."

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator
        self.instance_filter = instance_filter
        self.instance_comparator = instance_comparator
        self.traverse_time = traverse_time
        self.accumulate_number = accumulate_number
        self.mode = mode
        self.random_state = random.getstate()

    @property
    def instances(self):
        for dataset in load_data_objects(self.dataset_path):
            if self.mode == 'preserve':
                instances = dataset
            elif self.mode == 'ascend':
                instances = sorted(dataset, key=self.instance_comparator, reverse=False)
            elif self.mode == 'descend':
                instances = sorted(dataset, key=self.instance_comparator, reverse=True)
            elif self.mode == 'shuffle':
                instances = shuffled(dataset)

            for instance in instances:
                yield instance

    @property
    def batches(self):
        accum_batches = list()
        current_instances = list()
        max_size = 0
        for instance in self.instances:
            if self.instance_filter is not None and self.instance_filter(instance):
                continue
            current_instances.append(instance)
            max_size = max(max_size, self.instance_size_calculator(instance).max())
            total_size = len(current_instances) * max_size
            if total_size < self.batch_size:
                continue
            else:
                if total_size > self.batch_size:
                    accum_batches.append(Batch(instance.structure, current_instances[:-1]))
                    current_instances = current_instances[-1:]
                    max_size = self.instance_size_calculator(current_instances[-1]).max()
                else:
                    accum_batches.append(Batch(instance.structure, current_instances))
                    current_instances = list()
                    max_size = 0
            if len(accum_batches) == self.accumulate_number:
                yield accum_batches
                accum_batches = list()

    def __iter__(self):
        original_random_state = random.getstate()
        random.setstate(self.random_state)
        for _ in range(self.traverse_time):
            for batch in self.batches:
                if len(batch) == 0:
                    raise ValueError('Batch size too small, so batching no instance, please size up!')
                else:
                    yield batch

        random.setstate(original_random_state)
