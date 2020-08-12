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


from ynmt.trainers import Trainer

from ynmt.testers import build_tester
from ynmt.criterions import build_criterion

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.distributed import gather_all


def build_trainer_seq2seq(args,
                          model,
                          scheduler, optimizer,
                          vocabularies,
                          device_descriptor,
                          logger, visualizer):

    tester = build_tester(args.tester, vocabularies['target'])

    training_criterion = build_criterion(args.training_criterion, vocabularies['target'])
    validation_criterion = build_criterion(args.validation_criterion, vocabularies['target'])

    training_criterion.to(device_descriptor)
    validation_criterion.to(device_descriptor)

    seq2seq = Seq2Seq(
        args.name,
        model,
        scheduler, optimizer,
        tester,
        training_criterion, validation_criterion,
        vocabularies,
        args.normalization_type,
        args.checkpoint.directory,
        args.checkpoint.name,
        args.checkpoint.keep_number,
        device_descriptor,
        logger, visualizer,
    )
    return seq2seq


class Seq2Seq(Trainer):
    def __init__(self,
                 name,
                 model,
                 scheduler, optimizer,
                 tester,
                 training_criterion, validation_criterion,
                 vocabularies,
                 normalization_type,
                 checkpoint_directory,
                 checkpoint_name,
                 checkpoint_keep_number,
                 device_descriptor,
                 logger, visualizer):
        super(Seq2Seq, self).__init__(name,
                                      model,
                                      optimizer, scheduler,
                                      vocabularies,
                                      normalization_type,
                                      checkpoint_directory,
                                      checkpoint_name,
                                      checkpoint_keep_number,
                                      device_descriptor,
                                      logger, visualizer)
        self.tester = tester
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion

    def customize_accum_batch(self, accum_batch):
        padded_batches = list()
        normalization = 0
        for batch in accum_batch:
            padded_batch = Batch(set(['source', 'target']))
            # source side
            padded_attributes, _ = pad_attribute(batch.source, self.vocabularies['source'].pad_index)
            padded_batch.source = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

            # target side
            padded_attributes, _ = pad_attribute(batch.target, self.vocabularies['target'].pad_index)
            padded_batch.target = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

            # accumulate batch
            padded_batches.append(padded_batch)

            # Calculate normalization
            if self.normalization_type == 'token':
                valid_token_number = padded_batch.target[:, 1:].ne(self.vocabularies['target'].pad_index).sum().item()
                normalization += valid_token_number
            elif self.normalization_type == 'sentence':
                normalization += len(padded_batch.target)

        return padded_batches, normalization

    def train_accum_batch(self, customized_accum_train_batch):
        padded_train_batches, normalization = customized_accum_train_batch
        normalization = sum(gather_all(normalization, self.device_descriptor))
        self.train_statistics.clear()
        for batch in padded_train_batches:

            target_input = batch.target[:, :-1]
            target_output = batch.target[:, 1:]

            logits, attention_weight = self.model(batch.source, target_input)
            loss = self.training_criterion(logits, target_output)
            self.train_statistics += self.training_criterion.statistics

            loss /= normalization
            loss.backward()

        return

    def validate_accum_batch(self, customized_accum_valid_batch):
        padded_valid_batches, normalization = customized_accum_valid_batch
        for batch in padded_valid_batches:

            target_input = batch.target[:, :-1]
            target_output = batch.target[:, 1:]

            logits, attention_weight = self.model(batch.source, target_input)
            loss = self.validation_criterion(logits, target_output)
            self.valid_statistics += self.validation_criterion.statistics

        return
