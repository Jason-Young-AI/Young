#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-29 18:33
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


from ynmt.generators import Generator

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute


def build_generator_seq2seq(args,
                            model,
                            vocabularies,
                            device_descriptor,
                            logger):

    tester = build_tester(args.tester, vocabularies['target'])

    seq2seq = Seq2Seq(
        args.name,
        model,
        tester,
        vocabularies,
        device_descriptor,
        logger,
    )
    return seq2seq


class Seq2Seq(Generator):
    def __init__(self,
                 name,
                 model,
                 tester,
                 vocabularies,
                 device_descriptor,
                 logger):
        super(Seq2Seq, self).__init__(name,
                                      model,
                                      vocabularies,
                                      device_descriptor,
                                      logger)
        self.tester = tester

    def customize_batch(self, batch):
        padded_batch = Batch(set(['source', 'target']))
        # source side
        padded_attributes, _ = pad_attribute(batch.source, self.vocabularies['source'].pad_index)
        padded_batch.source = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

        # target side
        padded_attributes, _ = pad_attribute(batch.target, self.vocabularies['target'].pad_index)
        padded_batch.target = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

        return padded_batch

    def generate_batch(self, customized_batch):
        self.statistics.clear()

        source = customized_batch.source
        source_mask = self.model.get_source_mask(source)
        codes = self.model.encoder(source, source_mask)

        hidden, cross_attention_weight = self.model.decoder()

        target = customized_batch.target

        return
