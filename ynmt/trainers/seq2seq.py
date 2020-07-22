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
from ynmt.utilities.statistics import perplexity


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

    def get_normalization(self, packed_batch):
        normalization = 0
        for batch in packed_batch:
            padded_instances, instance_lengths = batch.target
            if self.normalization_type == 'token':
                normalization += sum(instance_lengths) - len(instance_lengths)
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

    def pad_batch(self, batch_iterator):
        for batch in batch_iterator:
            # source side
            attributes = batch['source']
            pad_index = self.vocabularies['source'].pad_index
            (padded_attributes, attribute_lengths) = pad_attribute(attributes, pad_index)
            padded_attributes = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)
            padded_attributes = padded_attributes.transpose(0, 1)
            attribute_lengths = torch.tensor(attribute_lengths, dtype=torch.long, device=self.device_descriptor)
            batch['source'] = (padded_attributes, attribute_lengths)

            # target side
            attributes = batch['target']
            pad_index = self.vocabularies['target'].pad_index
            (padded_attributes, attribute_lengths) = pad_attribute(attributes, pad_index)
            padded_attributes = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)
            padded_attributes = padded_attributes.transpose(0, 1)
            attribute_lengths = torch.tensor(attribute_lengths, dtype=torch.long, device=self.device_descriptor)
            batch['target'] = (padded_attributes, attribute_lengths)

            yield batch

    def update_optimizer_learning_rate(self, current_step):
        self.current_learning_rate = self.scheduler.learning_rate(current_step)
        for parameter_group in self.optimizer.parameter_groups:
            parameter_group['lr'] = self.current_learning_rate

    def train(self, train_batch_iterator):
        normalization = self.get_normalization(train_batch_iterator)
        for index, batch in enumerate(train_batch_iterator):
            source, source_lengths = batch.source
            target, target_lengths = batch.target

            target_input, target_output, target_lengths = self.get_decoder_io(target, target_lengths)

            prediction = self.model(source, target_input, source_lengths, target_lengths)
            loss, criterion_states = self.training_criterion(prediction, target_output, reduction='sum')
            loss = loss / normalization
            perp = perplexity(loss)
            loss_data = loss.item()
            print(loss_data)
            print(perp)
            loss.backward()
            return loss_data

    def validate(self, valid_batch_iterator):
        for index, batch in enumerate(valid_batch_iterator):
            source, source_lengths = batch.source
            target, target_lengths = batch.target

            target_input, target_output, target_lengths = self.get_decoder_io(target, target_lengths)

            prediction = self.model(source, target_input, source_lengths, target_lengths)
            loss, criterion_states = self.validation_criterion(prediction, target_output, reduction='sum')
