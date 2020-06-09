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


from ynmt.utilities.random import shuffled, random_state
from ynmt.data.batch import Batch


class Iterator(object):
    def __init__(self, dataset, batch_size, instance_size_calculator, traverse_time, mode):
        assert mode in {'preserve', 'sort', 'shuffle'}, "Wrong choice of order."

        self.dataset = dataset
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator
        self.traverse_time = traverse_time
        self.mode = mode
        self.iterations = 0
        self.rounds = 0
        self.random_state = random.getstate()

    @property
    def instances(self):
        random.setstate(self.random_state)
        if self.mode == 'preserve':
            instances = self.dataset
        elif self.mode == 'sort':
            instances = sorted(self.dataset)
        elif self.mode == 'shuffle':
            instances = shuffled(self.dataset)
        return instances

    @property
    def batches(self):
        current_instances = list()
        current_instances_size = 0
        for instance in self.instances:
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
