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


class Criterion(torch.nn.Module):
    def __init__(self, label_number, ignore_index):
        super(Criterion, self).__init__()
        self.label_number = label_number
        self.ignore_index = ignore_index

    def forward(self, predicted_distribution, ground_truth, reduction='none'):
        assert predicted_distribution.dim()-1 == ground_truth.dim()
        assert predicted_distribution.size()[:-1] == ground_truth.size()
        assert predicted_distribution.size(-1) == self.label_number

        ground_truth = ground_truth.unsqueeze(-1)

        mask = torch.eq(ground_truth, self.ignore_index)

        loss, states = self.compute_loss(predicted_distribution, ground_truth, mask)

        if reduction == 'sum':
            loss = torch.sum(loss)
        if reduction == 'mean':
            loss = torch.mean(loss)

        return loss, states

    def compute_loss(predicted_distribution, ground_truth, mask):
        raise NotImplementedError
