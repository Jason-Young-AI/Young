#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:03
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

from yoolkit.statistics import Statistics


class Criterion(torch.nn.Module):
    def __init__(self, label_number, ignore_index=-1):
        super(Criterion, self).__init__()
        self.label_number = label_number
        self.ignore_index = ignore_index
        self.statistics = Statistics(set())

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
