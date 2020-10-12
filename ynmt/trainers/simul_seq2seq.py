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

from ynmt.trainers import register_trainer, Trainer

from ynmt.criterions import build_criterion

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.statistics import perplexity
from ynmt.utilities.distributed import gather_all


@register_trainer('simul_seq2seq')
class SimulSeq2Seq(Trainer):
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
            device_descriptor, logger, visualizer
        )
        self.wait_time = wait_time
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion

        self.normalization_type = normalization_type

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

    def customize_accumulated_batch(self, accumulated_batch):
        accumulated_padded_batch = list()
        normalization = 0

        for batch in accumulated_batch:
            padded_batch = Batch(set({'source', 'target'}))
            # source side
            padded_source_attribute, _ = pad_attribute(batch['source'], self.task.vocabularies['source'].pad_index)
            padded_batch['source'] = torch.tensor(padded_source_attribute, dtype=torch.long, device=self.device_descriptor)

            # target side
            padded_target_attribute, _ = pad_attribute(batch['target'], self.task.vocabularies['target'].pad_index)
            padded_batch['target'] = torch.tensor(padded_target_attribute, dtype=torch.long, device=self.device_descriptor)

            # accumulate batch
            accumulated_padded_batch.append(padded_batch)

            # Calculate normalization
            if self.normalization_type == 'token':
                valid_token_number = padded_batch.target[:, 1:].ne(self.task.vocabularies['target'].pad_index).sum().item()
                normalization += valid_token_number
            elif self.normalization_type == 'sentence':
                normalization += len(padded_batch.target)

        return accumulated_padded_batch, normalization

    def train_accumulated_batch(self, customized_accumulated_train_batch):
        accumulated_padded_train_batch, normalization = customized_accumulated_train_batch
        normalization = sum(gather_all(normalization, self.device_descriptor))
        self.train_statistics.clear()
        accumulate_number = len(accumulated_padded_train_batch)
        for index, padded_train_batch in enumerate(accumulated_padded_train_batch):

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
                loss.backward()

    def validate_accumulated_batch(self, customized_accumulated_valid_batch):
        accumulated_padded_valid_batch, normalization = customized_accumulated_valid_batch
        for padded_valid_batch in accumulated_padded_valid_batch:

            target_input = padded_valid_batch.target[:, :-1]
            target_output = padded_valid_batch.target[:, 1:]

            logits, attention_weight = self.model(padded_valid_batch.source, target_input)
            loss = self.validation_criterion(logits, target_output)
            self.valid_statistics += self.validation_criterion.statistics

    def report(self, handle_name, reduced_statistics, time_cost):
        loss = reduced_statistics['loss']
        correct_item = reduced_statistics['correct_item']
        total_item = reduced_statistics['total_item']

        loss_per_item = loss / total_item
        ppl = perplexity(loss_per_item)
        accuracy = correct_item / total_item * 100

        report_string = f'{handle_name}@{self.step}/{self.life_cycle} - '
        report_string += f'loss/item: {loss_per_item:4.2f}; '
        report_string += f'ppl: {ppl:4.2f}; '
        report_string += f'acc: {accuracy:4.2f}%; '
        report_string += f'lr: {self.learning_rate:g}; '
        report_string += f'{time_cost:.0f}s'
        self.logger.info(report_string)

        l_win_name = f'{handle_name}_loss_per_item'
        p_win_name = f'{handle_name}_perplexity'
        a_win_name = f'{handle_name}_accuracy'

        l_win_title = f'[{handle_name}]: loss/item'
        p_win_title = f'[{handle_name}]: perplexity'
        a_win_title = f'[{handle_name}]: accuracy'

        self.visualizer.visualize(
            'line', l_win_name, l_win_title,
            opts=dict(
                legend = ['loss/item'],
            ),
            X=[self.step], Y=[loss_per_item],
            update="append",
        )

        self.visualizer.visualize(
            'line', p_win_name, p_win_title,
            opts=dict(
                legend = ['perplexity'],
            ),
            X=[self.step], Y=[ppl],
            update="append",
        )

        self.visualizer.visualize(
            'line', a_win_name, a_win_title,
            opts=dict(
                legend = ['accuracy'],
            ),
            X=[self.step], Y=[accuracy],
            update="append",
        )
