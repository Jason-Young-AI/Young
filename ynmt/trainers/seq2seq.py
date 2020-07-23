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
from ynmt.utilities.distributed import gather_all
from ynmt.utilities.statistics import Statistics
from ynmt.utilities.extractor import get_match_item


def build_trainer_seq2seq(args,
                          model, training_criterion, validation_criterion,
                          tester,
                          scheduler, optimizer,
                          vocabularies,
                          device_descritpor):
    seq2seq = Seq2Seq(
        args.name,
        model,
        training_criterion, validation_criterion,
        tester,
        scheduler, optimizer,
        vocabularies,
        args.accumulate_number,
        args.normalization_type,
        device_descritpor,
    )
    return seq2seq


class Seq2Seq(Trainer):
    def __init__(self,
                 name,
                 model,
                 training_criterion, validation_criterion,
                 tester,
                 scheduler, optimizer,
                 vocabularies,
                 accumulate_number,
                 normalization_type,
                 device_descriptor):
        super(Seq2Seq, self).__init__(name,
                                      model,
                                      training_criterion, validation_criterion,
                                      tester,
                                      optimizer, scheduler,
                                      vocabularies,
                                      accumulate_number,
                                      normalization_type,
                                      device_descriptor)
        self.statistics = Statistics({'tgt_loss', 'tgt_total_item', 'tgt_correct_item'})

    def get_normalization(self, packed_padded_train_batch):
        normalization = 0
        for batch in packed_padded_train_batch:
            padded_instances, instance_lengths = batch.target
            if self.normalization_type == 'token':
                valid_token = padded_instances[1:].ne(self.vocabularies['target'].pad_index)
                normalization += torch.sum(valid_token)
            elif self.normalization_type == 'sentence':
                normalization += len(instance_lengths)

        normalization_list = gather_all(normalization, self.device_descriptor)
        normalization = sum(normalization_list)
        return normalization

    def get_decoder_io(self, target, target_lengths):
        target_input = target[:-1]
        target_output = target[1:]

        target_max_length = torch.max(target_lengths)
        if not (target_max_length < len(target)):
            target_lengths = torch.where(target_lengths > target_max_length - 1, target_max_length - 1, target_lengths)
        return target_input, target_output, target_lengths

    def update_statistics(self, loss, prediction, target_output):
        # Statistic
        tgt_pad_index = self.vocabularies['target'].pad_index
        match_item = get_match_item(prediction.max(dim=-1)[1], target_output, tgt_pad_index)

        self.statistics.tgt_loss += loss.item()
        self.statistics.tgt_total_item += target_output.ne(tgt_pad_index).sum().item()
        self.statistics.tgt_correct_item += match_item.sum().item()

    def pad_batch(self, batches):
        for batch in batches:
            # source side
            attributes = batch.source
            pad_index = self.vocabularies['source'].pad_index
            (padded_attributes, attribute_lengths) = pad_attribute(attributes, pad_index)
            padded_attributes = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)
            padded_attributes = padded_attributes.transpose(0, 1)
            attribute_lengths = torch.tensor(attribute_lengths, dtype=torch.long, device=self.device_descriptor)
            batch.source = (padded_attributes, attribute_lengths)

            # target side
            attributes = batch.target
            pad_index = self.vocabularies['target'].pad_index
            (padded_attributes, attribute_lengths) = pad_attribute(attributes, pad_index)
            padded_attributes = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)
            padded_attributes = padded_attributes.transpose(0, 1)
            attribute_lengths = torch.tensor(attribute_lengths, dtype=torch.long, device=self.device_descriptor)
            batch.target = (padded_attributes, attribute_lengths)

            yield batch

    def train(self, packed_padded_train_batch):
        normalization = self.get_normalization(packed_padded_train_batch)
        self.statistics.clear()
        for index, batch in enumerate(packed_padded_train_batch):
            source, source_lengths = batch.source
            target, target_lengths = batch.target

            target_input, target_output, target_lengths = self.get_decoder_io(target, target_lengths)

            prediction = self.model(source, target_input, source_lengths, target_lengths)
            loss, criterion_states = self.training_criterion(prediction, target_output, reduction='sum')

            self.update_statistics(loss, prediction, target_output)

            loss /= normalization
            loss.backward()
        return self.statistics

    def validate(self, padded_valid_batches):
        self.statistics.clear()
        for index, batch in enumerate(padded_valid_batches):
            source, source_lengths = batch.source
            target, target_lengths = batch.target

            target_input, target_output, target_lengths = self.get_decoder_io(target, target_lengths)

            prediction = self.model(source, target_input, source_lengths, target_lengths)
            loss, criterion_states = self.validation_criterion(prediction, target_output, reduction='sum')

            self.update_statistics(loss, prediction, target_output)

        return self.statistics
