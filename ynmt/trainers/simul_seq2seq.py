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

from ynmt.trainers import register_trainer
from ynmt.trainers.seq2seq import Seq2Seq

from ynmt.criterions import build_criterion

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.statistics import perplexity, Statistics
from ynmt.utilities.distributed import gather_all


@register_trainer('simul_seq2seq')
class SimulSeq2Seq(Seq2Seq):
    def __init__(self,
        life_cycle,
        task, model, scheduler, optimizer,
        checkpoint_directory, checkpoint_name, checkpoint_keep_number,
        training_period, validation_period,
        report_period,
        wait_time,
        training_criterion, validation_criterion,
        normalization_type,
        device_descriptor, logger, visualizer
    ):
        super(SimulSeq2Seq, self).__init__(
            life_cycle,
            task, model, scheduler, optimizer,
            checkpoint_directory, checkpoint_name, checkpoint_keep_number,
            training_period, validation_period,
            report_period,
            training_criterion, validation_criterion,
            normalization_type,
            device_descriptor, logger, visualizer
        )
        self.wait_time = wait_time

    @classmethod
    def setup(cls, args, task, model, scheduler, optimizer, device_descriptor, logger, visualizer):

        training_criterion = build_criterion(args.training_criterion, task)
        validation_criterion = build_criterion(args.validation_criterion, task)

        training_criterion.to(device_descriptor)
        validation_criterion.to(device_descriptor)

        seq2seq = cls(
            args.life_cycle,
            task, model, scheduler, optimizer,
            args.checkpoints.directory, args.checkpoints.name, args.checkpoints.keep_number,
            args.training_period, args.validation_period,
            args.report_period,
            args.wait_time,
            training_criterion, validation_criterion,
            args.normalization_type,
            device_descriptor, logger, visualizer,
        )
        return seq2seq

    def train_accumulated_batch(self, customized_accumulated_train_batch):
        accumulated_padded_train_batch, statistics = customized_accumulated_train_batch
        normalization = sum(gather_all(statistics.normalization, self.device_descriptor))
        self.train_statistics += statistics
        for padded_train_batch in accumulated_padded_train_batch:

            target_input = padded_train_batch.target[:, :-1]
            target_output = padded_train_batch.target[:, 1:]

            read_end_position = padded_train_batch.source.shape[1] - 1
            if self.wait_time == -1:
                read_start_position = read_end_position
            else:
                read_start_position = min(self.wait_time + 1, read_end_position) # +1 for the bos token. When wait_time is 0, first read bos token

            read_position = read_start_position
            write_position = 0
            while read_start_position <= read_position and read_position <= read_end_position:
                if write_position == target_output.shape[1]:
                    break
                if read_position == read_end_position:
                    write_position = target_output.shape[1] - 1

                partial_source = padded_train_batch.source[:, :read_position + 1]
                if read_position == read_end_position:
                    partial_target_input = target_input
                    partial_target_output = target_output
                else:
                    partial_target_input = target_input[:, :write_position + 1]
                    partial_target_output = target_output[:, :write_position + 1]

                logits, attention_weight = self.model(partial_source, partial_target_input)
                loss = self.training_criterion(logits[:, write_position:], partial_target_output[:, write_position:])
                self.train_statistics += self.training_criterion.statistics

                read_position += 1
                write_position += 1

                loss /= normalization
                self.optimizer.backward(loss)

    def validate_accumulated_batch(self, customized_accumulated_valid_batch):
        accumulated_padded_valid_batch, statistics = customized_accumulated_valid_batch
        self.valid_statistics += statistics
        for padded_valid_batch in accumulated_padded_valid_batch:

            target_input = padded_valid_batch.target[:, :-1]
            target_output = padded_valid_batch.target[:, 1:]

            read_end_position = padded_valid_batch.source.shape[1] - 1
            if self.wait_time == -1:
                read_start_position = read_end_position
            else:
                read_start_position = min(self.wait_time + 1, read_end_position) # +1 for the bos token. When wait_time is 0, first read bos token

            read_position = read_start_position
            write_position = 0
            while read_start_position <= read_position and read_position <= read_end_position:
                if write_position == target_output.shape[1]:
                    break
                if read_position == read_end_position:
                    write_position = target_output.shape[1] - 1

                partial_source = padded_valid_batch.source[:, :read_position + 1]
                if read_position == read_end_position:
                    partial_target_input = target_input
                    partial_target_output = target_output
                else:
                    partial_target_input = target_input[:, :write_position + 1]
                    partial_target_output = target_output[:, :write_position + 1]

                logits, attention_weight = self.model(partial_source, partial_target_input)
                loss = self.validation_criterion(logits[:, write_position:], partial_target_output[:, write_position:])
                self.valid_statistics += self.validation_criterion.statistics

                read_position += 1
                write_position += 1
