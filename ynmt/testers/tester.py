#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:56
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch

from yoolkit.cio import load_data, dump_data
from yoolkit.timer import Timer

from ynmt.utilities.distributed import gather_all


class Tester(object):
    def __init__(self, factory, model, output_directory, output_name, device_descriptor, logger):
        assert os.path.isdir(output_directory), f'#3 arg(\'output_directory\') output directory: \'{output_directory}\' does not exist!'
        self.factory = factory
        self.model = model

        self.output_directory = output_directory
        self.output_name = output_name
        self.output_basepath = os.path.join(self.output_directory, self.output_name)

        self.device_descriptor = device_descriptor
        self.logger = logger

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.timer = Timer()

    def launch(self, output_label, indexed_testing_batches):
        self.logger.info(f' + Run Tester ...')

        self.timer.reset()
        self.timer.launch()

        if self.rank == 0:
            self.initialize(output_label)

        self.test_indexed_batches(indexed_testing_batches)

        if self.rank == 0:
            self.report()

        self.timer.reset()

        self.logger.info(f'   Test complete.')

    def test_indexed_batches(self, indexed_testing_batches):
        batch_number = len(indexed_testing_batches)
        batch_number_list = gather_all(batch_number, self.device_descriptor)
        max_batch_number = max(batch_number_list)

        self.model.train(False)
        with torch.no_grad():
            for iteration in range(max_batch_number):
                self.logger.info(f'   Testing iteration-{iteration} ...')
                if iteration < batch_number:
                    index, batch = indexed_testing_batches[iteration]
                    result = self.test(self.customize_batch(batch))
                else:
                    index = None
                    result = None

                indexed_results = gather_all((index, result), self.device_descriptor, data_size=65536)
                if self.rank == 0:
                    self.output_indexed_batches(indexed_results)

    def output_indexed_batches(self, indexed_results):

        def get_indexed_results():
            for indexed_result in indexed_results:
                if indexed_result[0] is None:
                    continue
                else:
                    yield indexed_result

        indexed_results = sorted(
            get_indexed_results(),
            key=lambda indexed_result: indexed_result[0],
            reverse=False
        )

        for index, result in indexed_results:
            self.output(result)
        self.logger.info(f'   {len(indexed_results)} batch has been tested. (Take: {self.timer.elapsed_time:2.0f}s)')

    @classmethod
    def setup(cls, settings, factory, model, device_descriptor, logger):
        raise NotImplementedError

    def initialize(self, output_label):
        raise NotImplementedError

    def customize_batch(self, batch):
        raise NotImplementedError

    def test(self, customized_batch):
        raise NotImplementedError

    def output(self, result):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
