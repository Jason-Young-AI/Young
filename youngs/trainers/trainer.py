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


import os
import torch

from yoolkit.timer import Timer
from yoolkit.statistics import Statistics

from youngs.utilities.checkpoint import save_checkpoint
from youngs.utilities.distributed import reduce_all, gather_all


class Trainer(object):
    def __init__(self,
        life_cycle,
        factory, model, scheduler, optimizer, tester,
        checkpoint_directory, checkpoint_name, checkpoint_keep_number,
        training_period, validation_period, testing_period, report_period,
        device_descriptor, logger, visualizer,
    ):
        assert os.path.isdir(checkpoint_directory), f'#6 arg(\'checkpoint_directory\') checkpoint directory: \'{checkpoint_directory}\' does not exist!'
        self.life_cycle = life_cycle

        self.factory = factory
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.tester = tester

        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_name = checkpoint_name
        self.checkpoint_keep_number = checkpoint_keep_number

        self.training_period = training_period
        self.validation_period = validation_period
        self.testing_period = testing_period
        self.report_period = report_period

        self.device_descriptor = device_descriptor
        self.logger = logger
        self.visualizer = visualizer

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.training_statistics = Statistics(set())
        self.validation_statistics = Statistics(set())
        self.step = 0
        self.timer = Timer()
        self.branch_timer = Timer()

    @classmethod
    def setup(cls, settings, factory, model, scheduler, optimizer, tester, device_descriptor, logger, visualizer):
        raise NotImplementedError

    @property
    def learning_rate(self):
        return self.scheduler.learning_rate(self.step)

    def reduce_statistics(self, statistics):
        statistics_list = gather_all(statistics, self.device_descriptor)
        reduced_statistics = sum(statistics_list, Statistics(set()))
        return reduced_statistics

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

    def save(self):
        self.timer.standby()

        self.branch_timer.reset()
        self.branch_timer.launch()

        if self.rank == 0:
            checkpoint = dict(
                step = self.step,
                model_state = self.model.state_dict(),
                optimizer_state = self.optimizer.state_dict(),
                scheduler_state = self.scheduler.state_dict(),
                model_settings = self.model.settings
            )
            self.logger.info(f' + Saving checkpoint ... ')
            save_checkpoint(checkpoint, self.checkpoint_directory, self.checkpoint_name, self.checkpoint_keep_number)
            self.logger.info(
                f'   Saved checkpoint to \'{self.checkpoint_directory}\' at {self.step} steps. '
                f'(Take: {self.branch_timer.elapsed_time:2.0f}s)'
            )

        self.timer.restart()
        return

    def launch(self, accumulated_training_batches, accumulated_validation_batches, indexed_testing_batches):
        self.training_statistics.clear()
        self.timer.reset()
        self.timer.launch()

        self.optimizer.zero_grad()

        for accumulated_training_batch in accumulated_training_batches:
            if self.step >= self.life_cycle:
                break

            # train
            self.train(accumulated_training_batch)

            # validate
            if self.step % self.validation_period == 0:
                self.validate(accumulated_validation_batches)

            if self.step % self.testing_period == 0:
                self.test(indexed_testing_batches)

            # save
            if self.step % self.training_period == 0:
                self.save()

        if self.step % self.training_period != 0:
            self.save()

        return

    def train(self, accumulated_training_batch):
        self.model.train(True) # Set to training mode may take some time, but it can aviod wrong operation of subclasses.
        self.timer.lap()
        self.training_statistics.clear()
        self.train_accumulated_batch(self.customize_accumulated_batch(accumulated_training_batch))

        self.update()
        reduced_training_statistics = self.reduce_statistics(self.training_statistics)
        if self.step % self.report_period == 0:
            self.report('Train', reduced_training_statistics, self.timer.lap(), self.timer.elapsed_time)

    def validate(self, accumulated_validation_batches):
        self.model.train(False)
        self.timer.standby()

        self.validation_statistics.clear()
        self.branch_timer.reset()
        self.branch_timer.launch()

        with torch.no_grad():
            for accumulated_validation_batch in accumulated_validation_batches:
                self.validate_accumulated_batch(self.customize_accumulated_batch(accumulated_validation_batch))

        reduced_validation_statistics = self.reduce_statistics(self.validation_statistics)
        if self.step % self.report_period == 0:
            self.report('Validate', reduced_validation_statistics, self.branch_timer.lap(), self.branch_timer.elapsed_time)

        self.timer.restart()

    def test(self, indexed_testing_batches):
        self.timer.standby()
        self.tester.launch(f'step_{self.step}', indexed_testing_batches)
        self.timer.restart()

    def report(self, handle_name, reduced_statistics, step_time_cost, total_time_cost):
        raise NotImplementedError

    def customize_accumulated_batch(self, accumulated_batch):
        # accumulated_batch is a list
        raise NotImplementedError

    def train_accumulated_batch(self, customized_accumulated_training_batch):
        # customized_accumulated_training_batch is a user-definded type
        raise NotImplementedError

    def validate_accumulated_batch(self, customized_accumulated_validation_batch):
        # customized_accumulated_validation_batch is a user-definded type
        raise NotImplementedError
