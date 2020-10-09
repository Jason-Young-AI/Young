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

from ynmt.criterions import register_criterion, Criterion


@register_criterion('cross_entropy')
class CrossEntropy(Criterion):
    def __init__(self, label_number, ignore_index):
        super(CrossEntropy, self).__init__(label_number, ignore_index)
        self.nll_loss = torch.nn.NLLLoss(ignore_index=self.ignore_index, reduction='sum')

    @classmethod
    def setup(cls, args, task):
        vocabulary = task.vocabularies['target']

        return cls(
            len(vocabulary),
            vocabulary.pad_index
        )

    def compute_loss(self, logits, ground_truth, valid_mask):
        # logits: [Batch_Size * Target_Length x Label_Number]
        # ground_truth: [Batch_Size * Target_Length]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        loss = self.nll_loss(log_probs, ground_truth)

        correct_item = log_probs.max(1)[1].eq(ground_truth).masked_select(valid_mask).sum().item()
        total_item = valid_mask.sum().item()

        # For average Loss & Accuracy & PPL
        self.statistics['loss'] = loss.item()
        self.statistics['correct_item'] = correct_item
        self.statistics['total_item'] = total_item

        return loss
