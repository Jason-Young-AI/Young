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


from ynmt.data.batch import pack_batch
from ynmt.utilities.timer import Timer
from ynmt.utilities.logging import get_logger
from ynmt.utilities.statistics import Statistics, perplexity
from ynmt.utilities.checkpoint import save_checkpoint
from ynmt.utilities.distributed import reduce_all, gather_all
from ynmt.utilities.visualizing import get_visualizer


class Trainer(object):
    def __init__(self,
                 name,
                 model,
                 training_criterion, validation_criterion,
                 tester,
                 optimizer, scheduler,
                 vocabularies,
                 accumulate_number,
                 normalization_type,
                 device_descriptor):
        self.name = name
        self.model = model
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion
        self.tester = tester
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocabularies = vocabularies
        self.accumulate_number = accumulate_number
        self.normalization_type = normalization_type
        self.device_descriptor = device_descriptor
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.step = 0
        self.timer = Timer()
        self.logger = get_logger('train')
        self.visualizer = get_visualizer(self.name)

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
                gradients.append(parameter.grad)
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
                report_string += f'{loss_name_prefix}loss: {loss:5.2f}; '
                report_statistics[f'{loss_name_prefix}loss'] = loss

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
                    accuracy = correct_item / gathered_statistics[total_item_name]
                    report_string += f'{correct_item_name_prefix}acc: {accuracy:4.2f}; '
                    report_statistics[f'{correct_item_name_prefix}acc'] = accuracy

                continue

            total_item_match_result = total_item_pattern.fullmatch(stat_name)
            if total_item_match_result is not None:
                continue

            if isinstance(stat_value, float):
                report_string += f'{stat_name}: {stat_value:4.2f}; '
            else:
                report_string += f'{stat_name}: {stat_value}; '

            report_statistics['stat_name'] = stat_value


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

    def launch(self, train_batches, training_period, valid_batches, validation_period,
               checkpoint_directory, checkpoint_name, checkpoint_keep_number):

        self.model.train(True)
        self.optimizer.zero_grad()

        padded_train_batches = self.pad_batch(train_batches)
        padded_valid_batches = self.pad_batch(valid_batches)
        packed_padded_train_batches = pack_batch(padded_train_batches, self.accumulate_number)

        self.timer.launch()

        total_statistics = Statistics(set())
        for index, packed_padded_train_batch in enumerate(packed_padded_train_batches):
            # train
            start_time = self.timer.elapsed_time
            self.timer.restart()
            train_statistics = self.train(packed_padded_train_batch)
            self.timer.standby()
            end_time = self.timer.elapsed_time
            self.update()
            train_report_statistics = self.report('Train', train_statistics, end_time - start_time)
            self.visualize('Train', train_report_statistics)
            total_statistics += train_statistics

            # save checkpoint
            if self.step % training_period == 0:
                checkpoint = dict(
                    step = self.step,
                    model = self.model.state_dict(),
                    optimizer = self.optimizer.state_dict(),
                    scheduler = self.scheduler.state_dict()
                )
                self.logger.info(f'Saving checkpoint ... ')
                start_time = self.timer.elapsed_time
                self.timer.restart()
                save_checkpoint(checkpoint_directory, checkpoint_name, checkpoint, checkpoint_keep_number)
                self.timer.standby()
                end_time = self.timer.elapsed_time
                self.logger.info(
                    f'Saved checkpoint to \'{checkpoint_directory}\' at {self.step} steps. '
                    f'(Cost: {end_time - start_time:6.0f}s)'
                )

            # validate
            if self.step % validation_period == 0:
                self.model.train(False)
                with torch.no_grad():
                    self.logger.info(f'Validating ... ')
                    start_time = self.timer.elapsed_time
                    self.timer.restart()
                    validate_status = self.validate(padded_valid_batches)
                    self.timer.standby()
                    end_time = self.timer.elapsed_time
                    valid_report_statistics = self.report('Validate', validate_status, end_time - start_time)
                    self.visualize('Validate', valid_report_statistics)
                self.model.train(True)

        self.report('Final', total_statistics, self.timer.elapsed_time)

    def pad_batch(self, batches):
        # batches is a generator
        raise NotImplementedError

    def train(self, packed_padded_train_batch):
        # packed_padded_train_batch is a list
        raise NotImplementedError

    def validate(self, padded_valid_batches):
        # padded_valid_batches is a generator
        raise NotImplementedError
