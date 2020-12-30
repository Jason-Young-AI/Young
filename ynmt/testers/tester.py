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

from yoolkit.cio import mk_temp, rm_temp, load_data, dump_data, dump_datas
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

        self.timer = Timer()

    def launch(self, output_label, indexed_testing_batches):
        self.rank = torch.distributed.get_rank()

        indexed_testing_batches = iter(indexed_testing_batches)
        self.logger.info(f' + Run Tester ...')
        self.result_path = mk_temp('ynmt-tester_', temp_type='file')

        self.timer.reset()
        self.timer.launch()

        if self.rank == 0:
            self.initialize(output_label)

        total_instance_number = 0
        while True:
            try:
                index, testing_batch = next(indexed_testing_batches)
                instance_number = len(testing_batch)
            except:
                index, testing_batch = None, None
                instance_number = 0

            if testing_batch is None:
                result = None
            else:
                result = self.test(testing_batch)

            self.output(index, result)
            gathered_instance_number = gather_all(instance_number, self.device_descriptor)
            instance_number = sum(gathered_instance_number)

            if instance_number == 0:
                break
            else:
                total_instance_number += instance_number
                self.logger.info(f'   {total_instance_number} instances has been tested. (Take: {self.timer.elapsed_time:2.0f}s)')

        if self.rank == 0:
            self.report()

        self.timer.reset()

        rm_temp(self.result_path)
        self.logger.info(f'   Test complete.')

    def test(self, testing_batch):
        self.model.train(False)
        with torch.no_grad():
            result = self.test_batch(self.customize_batch(testing_batch))
        return result

    def output(self, index, result):
        dump_data(self.result_path, (index, result))

        gathered_result_path = gather_all(self.result_path, self.device_descriptor)

        if self.rank == 0:
            indexed_results = list()
            for result_path in gathered_result_path:
                index, result = load_data(result_path)
                if index is None:
                    continue
                else:
                    indexed_results.append((index, result))

            indexed_results = sorted(indexed_results, key=lambda x: x[0])
            for index, result in indexed_results:
                self.output_result(result)

    @classmethod
    def setup(cls, settings, factory, model, device_descriptor, logger):
        raise NotImplementedError

    def initialize(self, output_label):
        raise NotImplementedError

    def customize_batch(self, batch):
        raise NotImplementedError

    def test_batch(self, customized_testing_batch):
        raise NotImplementedError

    def output_result(self, result):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
