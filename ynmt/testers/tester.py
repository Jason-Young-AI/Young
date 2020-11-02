#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:56
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

from yoolkit.timer import Timer


class Tester(object):
    def __init__(self, task, output_names, device_descriptor, logger):
        self.task = task
        self.output_names = output_names

        self.device_descriptor = device_descriptor
        self.logger = logger

        self.timer = Timer()

    def launch(self, model, output_basepath):
        self.timer.launch()

        for output_name in self.output_names:
            with open(output_basepath + '.' + output_name, 'w', encoding='utf-8') as output_file:
                output_file.truncate()

        model.train(False)
        with torch.no_grad():
            for index, batch in enumerate(self.input()):
                self.test(model, batch)
                self.output(output_basepath)
                self.logger.info(f'The No.{index} batch has been tested. (Take: {self.timer.elapsed_time:2.0f}s)')

        self.report(output_basepath)

        self.timer.reset()

    @classmethod
    def setup(cls, args, task, device_descriptor, logger):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def test(self, model, batch):
        raise NotImplementedError

    def input(self):
        raise NotImplementedError

    def output(self, output_basepath):
        raise NotImplementedError

    def report(self, output_basepath):
        raise NotImplementedError
