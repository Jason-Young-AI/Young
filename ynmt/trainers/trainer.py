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
from ynmt.utilities.checkpoint import save_checkpoint
from ynmt.utilities.distributed import reduce_all, gather_all


class Trainer(object):
    def __init__(self,
                 name,
                 model,
                 optimizer, scheduler,
                 vocabularies,
                 normalization_type,
                 checkpoint_directory,
                 checkpoint_name,
                 checkpoint_keep_number,
                 device_descriptor,
                 logger, visualizer):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocabularies = vocabularies
        self.normalization_type = normalization_type
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_name = checkpoint_name
        self.checkpoint_keep_number = checkpoint_keep_number
        self.device_descriptor = device_descriptor
        self.logger = logger
        self.visualizer = visualizer

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.train_statistics = Statistics(set())
        self.valid_statistics = Statistics(set())
        self.step = 0
        self.timer = Timer()
        self.branch_timer = Timer()

    @property
    def learning_rate(self):
        return self.scheduler.learning_rate(self.step)

    def update(self):
        self.step += 1

        # Reduce all gradients
        gradients = list()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.is_leaf:
                assert parameter.grad is not None, f'Parameter {name}: No gradient!'
                gradients.append(parameter.grad.data)
        reduce_all(gradients, self.device_descriptor)

        # Update learning rate
        for parameter_group in self.optimizer.parameter_groups:
            parameter_group['lr'] = self.learning_rate

        # Add all gradients
        self.optimizer.step()

        # Clear all gradients
        self.optimizer.zero_grad()

    def report(self, name, statistics, time_cost):
        report_string = f'{name}@{self.step} - '

        statistics_list = gather_all(statistics, self.device_descriptor)
        gathered_statistics = sum(statistics_list, Statistics(set()))

        loss_pattern = re.compile('(.*)loss')
        total_item_pattern = re.compile('(.*)total_item')
        correct_item_pattern = re.compile('(.*)correct_item')

        report_statistics = Statistics(set())
        for index, (stat_name, stat_value) in enumerate(gathered_statistics):
            loss_match_result = loss_pattern.fullmatch(stat_name)
            if loss_match_result is not None:
                loss_name_prefix = loss_match_result.group(1)
                loss = stat_value

                total_item_name = f'{loss_name_prefix}total_item'
                if total_item_name in gathered_statistics:
                    loss_per_item = loss / gathered_statistics[total_item_name]
                    report_string += f'{loss_name_prefix}loss/item: {loss_per_item:4.2f}; '
                    report_statistics[f'{loss_name_prefix}loss/item'] = loss_per_item
                    ppl = perplexity(loss_per_item)
                    report_string += f'{loss_name_prefix}ppl: {ppl:4.2f}; '
                    report_statistics[f'{loss_name_prefix}ppl'] = ppl

                continue

            correct_item_match_result = correct_item_pattern.fullmatch(stat_name)
            if correct_item_match_result is not None:
                correct_item_name_prefix = correct_item_match_result.group(1)
                correct_item = stat_value

                total_item_name = f'{correct_item_name_prefix}total_item'
                if total_item_name in gathered_statistics:
                    accuracy = correct_item / gathered_statistics[total_item_name] * 100
                    report_string += f'{correct_item_name_prefix}acc: {accuracy:4.2f}%; '
                    report_statistics[f'{correct_item_name_prefix}acc'] = accuracy

                continue

            total_item_match_result = total_item_pattern.fullmatch(stat_name)
            if total_item_match_result is not None:
                continue

            if isinstance(stat_value, float):
                report_string += f'{stat_name}: {stat_value:4.2f}; '
            else:
                report_string += f'{stat_name}: {stat_value}; '

            report_statistics[f'{stat_name}'] = stat_value


        report_string += f'lr: {self.learning_rate:g}; '
        report_string += f'{time_cost:.0f}s'
        self.logger.info(report_string)
        return report_statistics

    def visualize(self, name, report_statistics):
        for stat_name, stat_value in report_statistics:
            options = dict(
                legend = [stat_name]
            )
            stat_name = name + '_' + stat_name
            self.visualizer.visualize('line', stat_name, stat_name, opts=options, X=[self.step], Y=[stat_value], update="append")

    def launch(self, accum_train_batches, training_period, accum_valid_batches, validation_period):
        self.train_statistics.clear()
        self.model.train(True)
        self.optimizer.zero_grad()

        self.timer.launch()

        for index, accum_train_batch in enumerate(accum_train_batches):
            if index < self.step:
                continue

            # train
            self.train_accum_batch(self.customize_accum_batch(accum_train_batch))
            self.update()
            train_report_statistics = self.report('Train', self.train_statistics, self.timer.elapsed_time)
            self.visualize('Train', train_report_statistics)

            # validate
            if self.step % validation_period == 0:
                self.validate(accum_valid_batches)

            # save
            if self.step % training_period == 0:
                self.save()

        self.save()
        return

    def save(self):
        self.timer.standby()
        self.branch_timer.reset()
        self.branch_timer.launch()

        if self.rank == 0:
            checkpoint = dict(
                step = self.step,
                model = self.model.state_dict(),
                optimizer = self.optimizer.state_dict(),
                scheduler = self.scheduler.state_dict(),
                vocabularies = self.vocabularies,
                model_settings = self.model.settings
            )
            self.logger.info(f'Saving checkpoint ... ')
            save_checkpoint(checkpoint, self.checkpoint_directory, self.checkpoint_name, self.checkpoint_keep_number)
            self.logger.info(
                f'Saved checkpoint to \'{self.checkpoint_directory}\' at {self.step} steps. '
                f'(Cost: {self.branch_timer.elapsed_time:2.0f}s)'
            )

        self.timer.restart()
        return

    def validate(self, accum_valid_batches):
        self.timer.standby()

        self.branch_timer.reset()
        self.branch_timer.launch()

        self.valid_statistics.clear()
        self.model.train(False)
        with torch.no_grad():
            for accum_valid_batch in accum_valid_batches:
                self.validate_accum_batch(self.customize_accum_batch(accum_valid_batch))
        self.model.train(True)
        valid_report_statistics = self.report('Validate', self.valid_statistics, self.branch_timer.elapsed_time)
        self.visualize('Validate', valid_report_statistics)

        self.timer.restart()
        return

    def customize_accum_batch(self, accum_batch):
        # batch is a list
        raise NotImplementedError

    def train_accum_batch(self, customized_accum_train_batch):
        # padded_train_batch is a user-definded type
        raise NotImplementedError

    def validate_accum_batch(self, customized_accum_valid_batch):
        # padded_valid_batch is a user-definded type
        raise NotImplementedError
