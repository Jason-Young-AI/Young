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
from ynmt.data.attribute import pad_attribute
from ynmt.data.batch import Batch
from ynmt.utilities.distributed import gather_all
from ynmt.utilities.extractor import get_padding_mask, get_future_mask


def build_trainer_seq2seq(args,
                          model, model_settings,
                          training_criterion, validation_criterion,
                          tester,
                          scheduler, optimizer,
                          vocabularies,
                          device_descritpor):
    seq2seq = Seq2Seq(
        args.name,
        model, model_settings,
        training_criterion, validation_criterion,
        tester,
        scheduler, optimizer,
        vocabularies,
        args.normalization_type,
        device_descritpor,
    )
    return seq2seq


class Seq2Seq(Trainer):
    def __init__(self,
                 name,
                 model, model_settings,
                 training_criterion, validation_criterion,
                 tester,
                 scheduler, optimizer,
                 vocabularies,
                 normalization_type,
                 device_descriptor):
        super(Seq2Seq, self).__init__(name,
                                      model, model_settings,
                                      training_criterion, validation_criterion,
                                      tester,
                                      optimizer, scheduler,
                                      vocabularies,
                                      normalization_type,
                                      device_descriptor)

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
        bkn = normalization
        normalization = sum(gather_all(normalization, self.device_descriptor))
        self.train_statistics.clear()
        for batch in padded_train_batches:

            target_input = batch.target[:, :-1]
            target_output = batch.target[:, 1:]

            target_pad_index = self.vocabularies['target'].pad_index
            source_pad_index = self.vocabularies['source'].pad_index
            source_mask = get_padding_mask(batch.source, source_pad_index).unsqueeze(1)
            target_mask = get_padding_mask(target_input, target_pad_index).unsqueeze(1)
            target_mask = target_mask | get_future_mask(target_input).unsqueeze(0)

            logits, attention_weight = self.model(batch.source, target_input, source_mask, target_mask)

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

            target_pad_index = self.vocabularies['target'].pad_index
            source_pad_index = self.vocabularies['source'].pad_index
            source_mask = get_padding_mask(batch.source, source_pad_index).unsqueeze(1)
            target_mask = get_padding_mask(target_input, target_pad_index).unsqueeze(1)
            target_mask = target_mask | get_future_mask(target_input).unsqueeze(0)

            logits, attention_weight = self.model(batch.source, target_input, source_mask, target_mask)
            loss = self.validation_criterion(logits, target_output)
            self.valid_statistics += self.validation_criterion.statistics

        return
