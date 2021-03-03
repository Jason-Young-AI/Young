#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:09
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

from yoolkit.statistics import Statistics

from youngs.trainers import register_trainer, Trainer

from youngs.criterions import CrossEntropy, LabelSmoothingCrossEntropy

from youngs.data.batch import Batch
from youngs.data.attribute import pad_attribute

from youngs.utilities.metrics import perplexity
from youngs.utilities.distributed import gather_all


@register_trainer('fast_wait_k')
class FastWaitK(Trainer):
    def __init__(self,
        life_cycle,
        factory, model, scheduler, optimizer, tester,
        checkpoint_directory, checkpoint_name, checkpoint_keep_number,
        training_period, validation_period, testing_period, report_period,
        wait_source_time,
        training_criterion, validation_criterion,
        normalization_type,
        device_descriptor, logger, visualizer
    ):
        super(FastWaitK, self).__init__(
            life_cycle,
            factory, model, scheduler, optimizer, tester,
            checkpoint_directory, checkpoint_name, checkpoint_keep_number,
            training_period, validation_period, testing_period, report_period,
            device_descriptor, logger, visualizer
        )
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion

        self.normalization_type = normalization_type

        self.wait_source_time = wait_source_time

    @classmethod
    def setup(cls, settings, factory, model, scheduler, optimizer, tester, device_descriptor, logger, visualizer):
        args = settings.args

        training_criterion = LabelSmoothingCrossEntropy(
            len(factory.vocabularies['target']),
            args.label_smoothing_percent,
            factory.vocabularies['target'].pad_index
        )
        validation_criterion = CrossEntropy(
            len(factory.vocabularies['target']),
            factory.vocabularies['target'].pad_index
        )

        training_criterion.to(device_descriptor)
        validation_criterion.to(device_descriptor)

        trainer = cls(
            args.life_cycle,
            factory, model, scheduler, optimizer, tester,
            args.checkpoints.directory, args.checkpoints.name, args.checkpoints.keep_number,
            args.training_period, args.validation_period, args.testing_period, args.report_period,
            args.wait_source_time,
            training_criterion, validation_criterion,
            args.normalization_type,
            device_descriptor, logger, visualizer,
        )
        return trainer

    def customize_accumulated_batch(self, accumulated_batch):
        accumulated_padded_batch = list()
        statistics = Statistics(set({'normalization', 'src_token_number', 'tgt_token_number'}))
        for batch in accumulated_batch:
            # source side
            padded_source_attribute, _ = pad_attribute(batch.source, self.factory.vocabularies['source'].pad_index)
            source = torch.tensor(padded_source_attribute, dtype=torch.long, device=self.device_descriptor)

            # target side
            padded_target_attribute, _ = pad_attribute(batch.target, self.factory.vocabularies['target'].pad_index)
            target = torch.tensor(padded_target_attribute, dtype=torch.long, device=self.device_descriptor)

            # accumulate batch
            accumulated_padded_batch.append((source, target))

            statistics.src_token_number += source.ne(self.factory.vocabularies['source'].pad_index).sum().item()
            statistics.tgt_token_number += target.ne(self.factory.vocabularies['target'].pad_index).sum().item()
            # Calculate normalization
            if self.normalization_type == 'token':
                statistics.normalization += target[:, 1:].ne(self.factory.vocabularies['target'].pad_index).sum().item()
            elif self.normalization_type == 'sentence':
                statistics.normalization += len(target)

        return accumulated_padded_batch, statistics

    def train_accumulated_batch(self, customized_accumulated_training_batch):
        accumulated_padded_training_batch, statistics = customized_accumulated_training_batch
        normalization = sum(gather_all(statistics.normalization, self.device_descriptor))
        self.training_statistics += statistics

        for padded_training_batch in accumulated_padded_training_batch:
            source, target = padded_training_batch

            target_input = target[:, :-1]
            target_output = target[:, 1:]

            logits, cross_attention_weight = self.model(source, target_input, self.wait_source_time)
            loss = self.training_criterion(logits, target_output)
            self.training_statistics += self.training_criterion.statistics

            loss /= normalization
            self.optimizer.backward(loss)

    def validate_accumulated_batch(self, customized_accumulated_validation_batch):
        accumulated_padded_validation_batch, statistics = customized_accumulated_validation_batch
        self.validation_statistics += statistics
        for padded_validation_batch in accumulated_padded_validation_batch:
            source, target = padded_validation_batch

            target_input = target[:, :-1]
            target_output = target[:, 1:]

            logits, cross_attention_weight = self.model(source, target_input, self.wait_source_time)
            loss = self.validation_criterion(logits, target_output)
            self.validation_statistics += self.validation_criterion.statistics

    def report(self, handle_name, reduced_statistics, step_time_cost, total_time_cost):
        loss = reduced_statistics['loss']
        correct_item = reduced_statistics['correct_item']
        total_item = reduced_statistics['total_item']
        src_token_number = reduced_statistics['src_token_number']
        tgt_token_number = reduced_statistics['tgt_token_number']

        loss_per_item = loss / total_item
        ppl = perplexity(loss_per_item)
        accuracy = correct_item / total_item * 100
        src_tps = src_token_number / (step_time_cost + 1e-5)
        tgt_tps = tgt_token_number / (step_time_cost + 1e-5)

        report_string = f'{handle_name}@{self.step}/{self.life_cycle} - '
        report_string += f'loss/item: {loss_per_item:4.2f}; '
        report_string += f'ppl: {ppl:4.2f}; '
        report_string += f'acc: {accuracy:4.2f}%; '
        report_string += f'lr: {self.learning_rate:g}; '
        report_string += f'i/s: (s:{src_tps:.0f}|t:{tgt_tps:.0f}); '
        report_string += f'{total_time_cost:.0f}sec'
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
