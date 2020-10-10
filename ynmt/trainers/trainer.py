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


import torch

from ynmt.utilities.timer import Timer
from ynmt.utilities.statistics import Statistics
from ynmt.utilities.checkpoint import save_checkpoint
from ynmt.utilities.distributed import reduce_all, gather_all


class Trainer(object):
    def __init__(self,
        life_cycle,
        task, model, scheduler, optimizer,
        checkpoint_directory, checkpoint_name, checkpoint_keep_number,
        training_period, validation_period,
        report_period,
        device_descriptor, logger, visualizer,
    ):
        self.life_cycle = life_cycle
        self.task = task
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_name = checkpoint_name
        self.checkpoint_keep_number = checkpoint_keep_number
        self.training_period = training_period
        self.validation_period = validation_period
        self.report_period = report_period
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
    def setup(cls, args, task, model, scheduler, optimizer, device_descriptor, logger, visualizer):
        raise NotImplementedError

    @property
    def learning_rate(self):
        return self.scheduler.learning_rate(self.step)

    def reduce_statistics(self, statistics):
        statistics_list = gather_all(statistics, self.device_descriptor)
        reduced_statistics = sum(statistics_list, Statistics())
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
                model = self.model.state_dict(),
                optimizer = self.optimizer.state_dict(),
                scheduler = self.scheduler.state_dict(),
                model_args = self.model.args
            )
            self.logger.info(f'Saving checkpoint ... ')
            save_checkpoint(checkpoint, self.checkpoint_directory, self.checkpoint_name, self.checkpoint_keep_number)
            self.logger.info(
                f'Saved checkpoint to \'{self.checkpoint_directory}\' at {self.step} steps. '
                f'(Take: {self.branch_timer.elapsed_time:2.0f}s)'
            )

        self.timer.restart()
        return

    def launch(self, accumulated_train_batches, accumulated_valid_batches):
        self.train_statistics.clear()
        self.timer.reset()
        self.timer.launch()

        self.optimizer.zero_grad()

        for accumulated_train_batch in accumulated_train_batches:
            if self.step >= self.life_cycle:
                break

            # train
            self.model.train(True) # Set to training mode may take some time, but it can aviod wrong operation of subclasses.
            self.train(accumulated_train_batch)

            # validate
            if self.step % self.validation_period == 0:
                self.model.train(False)
                self.validate(accumulated_valid_batches)

            # save
            if self.step % self.training_period == 0:
                self.save()

        if self.step % self.training_period != 0:
            self.save()

        return

    def train(self, accumulated_train_batch):
        self.train_accumulated_batch(self.customize_accumulated_batch(accumulated_train_batch))

        self.update()
        reduced_train_statistics = self.reduce_statistics(self.train_statistics)
        if self.step % self.report_period == 0:
            self.report('Train', reduced_train_statistics, self.timer.elapsed_time)

    def validate(self, accumulated_valid_batches):
        self.timer.standby()

        self.valid_statistics.clear()
        self.branch_timer.reset()
        self.branch_timer.launch()

        with torch.no_grad():
            for accumulated_valid_batch in accumulated_valid_batches:
                self.validate_accumulated_batch(self.customize_accumulated_batch(accumulated_valid_batch))

        reduced_valid_statistics = self.reduce_statistics(self.valid_statistics)
        if self.step % self.report_period == 0:
            self.report('Validate', reduced_valid_statistics, self.branch_timer.elapsed_time)

        self.timer.restart()

    def report(self, handle_name, reduced_statistics, time_cost):
        raise NotImplementedError

    def customize_accumulated_batch(self, accumulated_batch):
        # accumulated_batch is a list
        raise NotImplementedError

    def train_accumulated_batch(self, customized_accumulated_train_batch):
        # customized_accumulated_train_batch is a user-definded type
        raise NotImplementedError

    def validate_accumulated_batch(self, customized_accumulated_valid_batch):
        # customized_accumulated_valid_batch is a user-definded type
        raise NotImplementedError
