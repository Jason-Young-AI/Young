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


from ynmt.utilities.checkpoint import save_checkpoint


class Trainer(object):
    def __init__(self, model,
                 training_criterion, validation_criterion,
                 tester,
                 optimizer, scheduler,
                 accumulate_number,
                 normalization_type,
                 device_descriptor):
        self.model = model
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion
        self.tester = tester
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accumulate_number = accumulate_number
        self.normalization_type = normalization_type
        self.device_descriptor = device_descriptor
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

    def launch(self, train_batches, training_period, valid_batches, validation_period,
               checkpoint_directory, checkpoint_name, checkpoint_keep_number):
        self.model.train(True)
        packed_batches = pack_multi_padded_batch(train_batches, self.accumulate_number)
        for index, packed_batch in enumerate(packed_batches):
            current_step = self.optimizer.current_step

            self.optimizer.zero_grad()
            self.train(packed_batch)
            self.reduce_all_gradients()
            self.update_optimizer_learning_rate(current_step)
            self.optimizer.step()

            if current_step % training_period == 0:
                checkpoint = dict()
                checkpoint['step'] = current_step
                checkpoint['model'] = self.model.state_dict()
                checkpoint['optimizer'] = self.optimizer.state_dict()
                save_checkpoint(checkpoint_directory, checkpoint_name, checkpoint, checkpoint_keep_number)

            if current_step % validation_period == 0:
                self.model.train(False)
                with torch.no_grad():
                    self.validate(valid_batches)
                self.model.train(True)

    def reduce_all_gradients(self):
        gradients = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.is_leaf:
                assert parameter.grad is not None, f'Parameter {name}: No gradient!')
                gradients.append(parameter.grad)
        reduce_all(gradients)

    def update_optimizer_learning_rate(self, current_step):
        raise NotImplementedError

    def validate(self, valid_batches):
        raise NotImplementedError

    def train(self, packed_batch):
        raise NotImplementedError
