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


from ynmt.trainer import Trainer
from ynmt.data.batch import pack_multi_padded_batch
from ynmt.utilities.distributed import gather_all, reduce_all


def build_trainer_seq2seq(args, model, optimizer, is_station):
    pass


class Seq2Seq(Trainer):
    def __init__(self, model,
                 training_criterion, validation_criterion,
                 tester,
                 optimizer, scheduler,
                 accumulate_number,
                 normalization_type,
                 device_descriptor):
        super(Seq2Seq, self).__init__(model,
                                      training_criterion, validation_criterion,
                                      tester,
                                      optimizer, scheduler,
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

        normalization_list = gather_all(normalization)
        normalization = sum(normalization_list)
        return normalization

    def update_optimizer_learning_rate(self, current_step):
        learning_rate = self.scheduler.learning_rate(current_step)
        for parameter_group in self.optimizer.parameter_groups:
            parameter_group['lr'] = learning_rate

    def train(self, packed_batch):
        normalization = self.get_normalization(packed_batch)
        for index, batch in enumerate(packed_batch):
            source_padded_instances, source_instance_lengths = batch.source
            target_padded_instances, target_instance_lengths = batch.target
            prediction = self.model(source_padded_instances, target_padded_instances, source_instance_lengths, target_instance_lengths)
            loss = self.training_criterion(target_padded_instances, prediction)
            loss.backward()

    def validate(self, valid_batches):
        for index, batch in enumerate(valid_batches):
            source_padded_instances, source_instance_lengths = batch.source
            target_padded_instances, target_instance_lengths = batch.target
            prediction = self.model(source_padded_instances, target_padded_instances, source_instance_lengths, target_instance_lengths)
            loss = self.validation_criterion(target_padded_instances, prediction)
