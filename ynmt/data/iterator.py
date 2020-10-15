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
from ynmt.utilities.file import load_datas
from ynmt.utilities.random import shuffled
from ynmt.utilities.statistics import Statistics


class Iterator(object):
    def __init__(self,
        dataset_path, container_size, dock_size=1, export_volume=1,
        instance_filter=None, instance_comparator=None, instance_size_calculator=None,
        shuffle=False, infinite=False,
    ):
        self.dataset_path = dataset_path
        self.container_size = container_size
        self.dock_size = dock_size
        self.export_volume = export_volume

        self.instance_filter = instance_filter
        self.instance_comparator = instance_comparator
        self.instance_size_calculator = instance_size_calculator

        self.infinite = infinite

        self.shuffle = shuffle
        self.random_state = random.getstate()

    def pack_instances(self, instances, size):
        packed_instances = list()
        for instance in instances:
            packed_instances.append(instance)
            packed_instances_size = self.instance_size_calculator(packed_instances)

            if packed_instances_size > size:
                yield packed_instances[:-1]
                packed_instances = packed_instances[-1:]
                packed_instances_size = self.instance_size_calculator(packed_instances)

            if packed_instances_size == size:
                yield packed_instances
                packed_instances = list()
                packed_instances_size = 0

        if len(packed_instances) != 0:
            yield packed_instances

    @property
    def instances(self):
        for dataset in load_datas(self.dataset_path):
            if self.shuffle:
                dataset = shuffled(dataset)
            for index, instance in enumerate(dataset):
                if self.instance_filter is None or not self.instance_filter(instance):
                    yield instance

    @property
    def containers(self):
        docks = self.pack_instances(self.instances, self.container_size * self.dock_size)
        for dock in docks:
            if self.instance_comparator is not None and self.shuffle:
                dock = sorted(dock, key=self.instance_comparator)

            containers = self.pack_instances(dock, self.container_size)
            if self.shuffle:
                containers = shuffled(list(containers))

            for container in containers:
                assert len(container) != 0, f'Container size is too small, packed no instance, please size up!{container}'
                yield Batch(container[-1].structure, container)

    def __iter__(self):
        original_random_state = random.getstate()
        random.setstate(self.random_state)

        exports = list()
        while True:
            for container in self.containers:
                exports.append(container)
                if len(exports) == self.export_volume:
                    yield exports
                    exports = list()

            if not self.infinite:
                break

        random.setstate(original_random_state)


class RawTextIterator(object):
    def __init__(self, raw_text_paths, instance_handler, batch_size, batch_size_calculator):
        assert isinstance(raw_text_paths, list), f'raw_text_paths must be a list of path.'
        self.raw_text_paths = raw_text_paths
        self.instance_handler = instance_handler
        self.batch_size = batch_size
        self.batch_size_calculator = batch_size_calculator

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
            current_batch_size = self.batch_size_calculator(current_instances)

            if current_batch_size > self.batch_size:
                yield Batch(instance.structure, current_instances[:-1])
                current_instances = current_instances[-1:]
                current_batch_size = self.batch_size_calculator(current_instances)

            if current_batch_size == self.batch_size:
                yield Batch(instance.structure, current_instances)
                current_instances = list()
                current_batch_size = 0

        if len(current_instances) != 0:
            yield Batch(current_instances[-1].structure, current_instances)

    def __iter__(self):
        for batch in self.batches:
            if len(batch) == 0:
                raise ValueError('Batch size too small, so batching no instance, please size up!')
            else:
                yield batch
