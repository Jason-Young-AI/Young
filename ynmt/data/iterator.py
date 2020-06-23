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


from ynmt.data.batch import Batch
from ynmt.utilities.file import load_data_objects
from ynmt.utilities.random import shuffled, random_state


class Iterator(object):
    def __init__(self, dataset_path, batch_size, instance_size_calculator, instance_filter=None, traverse_time=1, mode='preserve'):
        assert mode in {'preserve', 'sort', 'shuffle'}, "Wrong choice of order."

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator
        self.instance_filter = instance_filter
        self.traverse_time = traverse_time
        self.mode = mode
        self.iterations = 0
        self.rounds = 0
        self.random_state = random.getstate()

    @property
    def instances(self):
        original_random_state = random.getstate()
        random.setstate(self.random_state)
        for dataset in load_data_objects(self.dataset_path):
            if self.mode == 'preserve':
                instances = dataset
            elif self.mode == 'sort':
                instances = sorted(dataset)
            elif self.mode == 'shuffle':
                instances = shuffled(dataset)
            yield instances

        random.setstate(original_random_state)

    @property
    def batches(self):
        current_instances = list()
        current_instances_size = 0
        for instance in self.instances:
            if self.instance_filter is not None and self.instance_filter(instance):
                continue
            current_instances.append(instance)
            current_instances_size += self.instance_size_calculator(instance)
            if current_instances_size < self.batch_size:
                continue
            else:
                if current_instances_size > self.batch_size:
                    yield Batch(self.dataset.structure, current_instances[:-1])
                    current_instances = current_instances[-1:]
                    current_instances_size = self.instance_size_calculator(current_instances[-1])
                else:
                    yield Batch(self.dataset.structure, current_instances)
                    current_instances = list()
                    current_instances_size = 0

    def __iter__(self):
        for time in range(self.traverse_time):
            if time < self.rounds:
                continue
            for index, batch in enumerate(self.batches)):
                if index < self.iterations:
                    continue
                if batch.size == 0:
                    raise ValueError('Batch size too small, so batching no instance, please size up!')
                else:
                    yield batch
                self.iterations += 1
            self.rounds += 1
