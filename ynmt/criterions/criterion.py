#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-30 09:38
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


from ynmt.utilities.statistics import Statistics


class Criterion(torch.nn.Module):
    def __init__(self, label_number, ignore_index):
        super(Criterion, self).__init__()
        self.label_number = label_number
        self.ignore_index = ignore_index
        self.statistics = Statistics(set())

    @classmethod
    def setup(cls, args, task):
        raise NotImplementedError

    def forward(self, logits, ground_truth):
        assert logits.dim() - 1 == ground_truth.dim(), f'Wrong number of dimension: #1 arg:{logits.size()}, #2 arg: {ground_truth.size()}'
        assert logits.size()[:-1] == ground_truth.size(), f'Wrong size: #1 arg:{logits.size()}, #2 arg: {ground_truth.size()}'
        assert logits.size(-1) == self.label_number, f'Number of label should be {self.label_number} instead: #1 arg:{logits.size(-1)}'

        logits = logits.reshape(-1, logits.size(-1))
        ground_truth = ground_truth.reshape(-1)

        valid_mask = ground_truth.ne(self.ignore_index)

        return self.compute_loss(logits, ground_truth, valid_mask)

    def compute_loss(logits, ground_truth, valid_mask):
        raise NotImplementedError
