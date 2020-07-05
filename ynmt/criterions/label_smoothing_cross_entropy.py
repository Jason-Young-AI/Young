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


def build_criterion_label_smoothing_cross_entropy(args, vocabulary):
    label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(
        len(vocabulary),
        args.label_smoothing_percent,
        vocabulary.pad_index
    )
    return label_smoothing_cross_entropy


class LabelSmoothingCrossEntropy(Criterion):
    def __init__(self, label_number, label_smoothing_percent, ignore_index):
        assert 0.0 < label_smoothing_percent and label_smoothing_percent < 1.0
        super(LabelSmoothingCrossEntropy, self).__init__(label_number, ignore_index)
        self.label_smoothing_percent = label_smoothing_percent


    def compute_loss(self, predicted_distribution, ground_truth, mask):
        log_predicted_distribution = -torch.log(predicted_distribution)

        # cross entropy between 'one-hot encoded distribution' & 'predicted_distribution'
        ce_op = torch.gather(log_predicted_distribution, dim=-1, index=ground_truth)
        ce_op.masked_fill_(mask, 0)

        # cross entropy between 'noise distribution' & 'predicted_distribution'
        ce_np = (1 / self.label_number) * torch.sum(log_predicted_distribution, dim=-1, keepdim=True)
        ce_np.masked_fill_(mask, 0)

        loss = (1 - self.label_smoothing_percent) * ce_op + self.label_smoothing_percent * ce_np

        return loss, (ce_op, ce_np)
