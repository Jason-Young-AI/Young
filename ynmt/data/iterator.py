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
from ynmt.data.instance import Instance, InstanceComparator
from ynmt.utilities.file import load_datas
from ynmt.utilities.random import shuffled
from ynmt.utilities.statistics import Statistics


class Iterator(object):
    def __init__(self,
        dataset_path, batch_size,
        instance_size_calculator, instance_filter=None, instance_comparator=InstanceComparator(),
        accumulate_number=1, mode='preserve', infinite=False,
    ):
        assert mode in {'preserve', 'ascend', 'descend', 'shuffle'}, "Wrong choice of order."

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator
        self.instance_filter = instance_filter
        self.instance_comparator = instance_comparator
        self.accumulate_number = accumulate_number
        self.mode = mode
        self.infinite = infinite
        self.random_state = random.getstate()

    @property
    def instances(self):
        for dataset in load_datas(self.dataset_path):
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

        if len(current_instances) != 0:
            yield [Batch(current_instances[-1].structure, current_instances)]

    def __iter__(self):
        original_random_state = random.getstate()
        random.setstate(self.random_state)
        while True:
            for batch in self.batches:
                if len(batch) == 0:
                    raise ValueError('Batch size too small, so batching no instance, please size up!')
                else:
                    yield batch
            if not self.infinite:
                break

        random.setstate(original_random_state)


class RawTextIterator(object):
    def __init__(self, raw_text_paths, instance_handler, batch_size, instance_size_calculator):
        assert isinstance(raw_text_paths, list), f'raw_text_paths must be a list of path.'
        self.raw_text_paths = raw_text_paths
        self.instance_handler = instance_handler
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator

    @property
    def instances(self):
        raw_text_files = list()
        for raw_text_path in self.raw_text_paths:
            raw_text_files.append(open(raw_text_path, 'r', encoding='utf-8'))

        for lines in zip(*raw_text_files):
            yield self.instance_handler(lines)

        for raw_text_file in raw_text_files:
            raw_text_file.close()

    @property
    def batches(self):
        current_instances = list()
        max_size = 0

        for instance in self.instances:
            current_instances.append(instance)
            max_size = max(max_size, self.instance_size_calculator(instance).max())
            total_size = len(current_instances) * max_size
            if total_size < self.batch_size:
                continue
            else:
                if total_size > self.batch_size:
                    yield Batch(instance.structure, current_instances[:-1])
                    current_instances = current_instances[-1:]
                    max_size = self.instance_size_calculator(current_instances[-1]).max()
                else:
                    yield Batch(instance.structure, current_instances)
                    current_instances = list()
                    max_size = 0
        if len(current_instances) != 0:
            yield Batch(current_instances[-1].structure, current_instances)

    def __iter__(self):
        for batch in self.batches:
            if len(batch) == 0:
                raise ValueError('Batch size too small, so batching no instance, please size up!')
            else:
                yield batch
