#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-30 09:39
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


from ynmt.criterions import Criterion


def build_criterion_cross_entropy(args, vocabulary):
    cross_entropy = CrossEntropy(
        len(vocabulary),
        vocabulary.pad_index
    )
    return cross_entropy


class CrossEntropy(Criterion):
    def compute_loss(self, predicted_distribution, ground_truth, mask):
        log_predicted_distribution = -torch.log(predicted_distribution)

        # cross entropy between 'one-hot encoded distribution' & 'predicted_distribution'
        loss = torch.gather(log_predicted_distribution, dim=-1, index=ground_truth)
        loss.masked_fill_(mask, 0)

        return loss, None
