#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import random

from yoolkit.cio import load_lines, load_datas
from yoolkit.statistics import Statistics

from youngs.data.batch import Batch
from youngs.data.dataset import Dataset
from youngs.data.instance import Instance
from youngs.utilities.random import shuffled


def build_batch(instances):
    batch = Batch(instances[0].structure)
    for instance in instances:
        batch.add(instance)
    return batch


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
            assert isinstance(dataset, Dataset)
            if self.shuffle:
                dataset = shuffled(dataset)
            for instance in dataset:
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
                assert len(container) != 0, f'Container size is too small, packed no instance, please size up!'
                yield build_batch(container)

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


class RawIterator(object):
    def __init__(self, raw_paths, raw_modes, instance_handler, batch_size, instance_size_calculator):
        assert isinstance(raw_paths, list), f'raw_paths must be a list of path.'
        assert isinstance(raw_modes, list), f'raw_modes must be a list of mode.'
        for raw_mode in raw_modes:
            assert raw_mode in {'text', 'binary'}

        self.raw_paths = raw_paths
        self.raw_modes = raw_modes
        self.instance_handler = instance_handler
        self.batch_size = batch_size
        self.instance_size_calculator = instance_size_calculator

    @property
    def instances(self):
        raw_files = list()
        for raw_path, raw_mode in zip(self.raw_paths, self.raw_modes):
            if raw_mode == 'text':
                raw_files.append(load_lines(raw_path))
            if raw_mode == 'binary':
                raw_files.append(load_datas(raw_path))

        for attributes in zip(*raw_files):
            instance = self.instance_handler(attributes)
            assert isinstance(instance, Instance)
            yield instance

    @property
    def batches(self):
        current_instances = list()
        max_size = 0

        for instance in self.instances:
            current_instances.append(instance)
            current_batch_size = self.instance_size_calculator(current_instances)

            if current_batch_size > self.batch_size:
                yield build_batch(current_instances[:-1])
                current_instances = current_instances[-1:]
                current_batch_size = self.instance_size_calculator(current_instances)

            if current_batch_size == self.batch_size:
                yield build_batch(current_instances)
                current_instances = list()
                current_batch_size = 0

        if len(current_instances) != 0:
            yield build_batch(current_instances)

    def __iter__(self):
        for index, batch in enumerate(self.batches):
            if len(batch) == 0:
                raise ValueError('Batch size too small, so batching no instance, please size up!')
            else:
                yield index, batch
