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

    def launch(self, indexed_testing_batches):
        self.timer.reset()
        self.timer.launch()

        iteration_number = len(indexed_testing_batches)
        gathered_iteration_number = gather_all(iteration_number, self.device_descriptor)
        max_iteration_number = max(gathered_iteration_number)

        self.logger.info(f'Run Tester ...')
        self.model.train(False)
        with torch.no_grad():
            for iteration in range(max_iteration_number):
                if iteration < len(indexed_testing_batches):
                    index, batch = indexed_testing_batches[iteration]
                    customized_batch = self.customize_batch(batch)
                    result = self.test(customized_batch)
                else:
                    result = None
                indexed_results = gather_all((index, result), self.device_descriptor, data_size=65535)
                indexed_results = sorted(indexed_results, key=lambda indexed_result: indexed_result[0], reverse=False)

                if self.rank == 0:
                    batch_number = 0
                    for index, result in indexed_results:
                        if result is None:
                            continue
                        else:
                            self.output(result)
                            batch_number += 1
                    self.logger.info(f'Iter-{iteration}: {batch_number} batch has been tested. (Take: {self.timer.elapsed_time:2.0f}s)')
            self.report()
        self.timer.reset()

    @classmethod
    def setup(cls, settings, factory, model, device_descriptor, logger):
        raise NotImplementedError

    def initialize(self, output_extension):
        raise NotImplementedError

    def customize_batch(self, batch):
        raise NotImplementedError

    def test(self, customized_batch):
        raise NotImplementedError

    def output(self, result):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
