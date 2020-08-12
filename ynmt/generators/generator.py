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


import re
import torch


from ynmt.utilities.timer import Timer
from ynmt.utilities.statistics import Statistics, perplexity


class Generator(object):
    def __init__(self,
                 name,
                 model,
                 vocabularies,
                 device_descriptor,
                 logger):
        self.name = name
        self.model = model
        self.vocabularies = vocabularies
        self.device_descriptor = device_descriptor
        self.logger = logger

        self.statistics = Statistics(set())
        self.timer = Timer()

   def launch(self, batches, output_paths):
        self.statistics.clear()
        self.timer.launch()

        self.model.train(False)
        with torch.no_grad():
            for index, batch in enumerate(batches):
                self.generate_batch(self.customize_batch(batch))

        return

    def customize_batch(self, batch):
        # batch is a object of Class Batch()
        raise NotImplementedError

    def generate_batch(self, customized_batch):
        # customized_batch is a object of Class Batch()
        raise NotImplementedError
