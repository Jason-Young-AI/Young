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


class LabelSmoothingCrossEntropy(Criterion):
    def __init__(self, label_number, label_smoothing_percent, ignore_index):
        assert 0.0 < label_smoothing_percent and label_smoothing_percent < 1.0

        super(LabelSmoothingCrossEntropy, self).__init__(label_number, ignore_index)

        smoothed_distribution = torch.full([label_number], label_smoothing_percent / (label_number - 2))
        smoothed_distribution[ignore_index] = 0
        self.register_buffer('smoothed_distribution', smoothed_distribution)

        self.label_smoothing_percent = label_smoothing_percent
        self.kl_div_loss = torch.nn.KLDivLoss(reduction='sum')

    def compute_loss(self, logits, ground_truth, valid_mask):
        # logits: [Batch_Size * Target_Length x Label_Number]
        # ground_truth: [Batch_Size * Target_Length]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        real_probs = self.smoothed_distribution.repeat(logits.size(0), 1)
        real_probs.scatter_(1, ground_truth.unsqueeze(1), 1 - self.label_smoothing_percent)
        real_probs.masked_fill_(~valid_mask.unsqueeze(1), 0)
        loss = self.kl_div_loss(log_probs, real_probs)

        correct_item = log_probs.max(1)[1].eq(ground_truth).masked_select(valid_mask).sum().item()
        total_item = valid_mask.sum().item()

        # For average Loss & Accuracy & PPL
        self.statistics['loss'] = loss.item()
        self.statistics['correct_item'] = correct_item
        self.statistics['total_item'] = total_item

        return loss
