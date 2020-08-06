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


from ynmt.trainers.trainer import Trainer
from ynmt.trainers.seq2seq import build_trainer_seq2seq


def build_trainer(args,
                  model, model_settings,
                  training_criterion, validation_criterion,
                  tester,
                  scheduler, optimizer,
                  vocabularies,
                  device_descritpor,
                  checkpoint, reset_step):
    trainer = globals()[f'build_trainer_{args.name}'](
        args,
        model, model_settings,
        training_criterion, validation_criterion,
        tester,
        scheduler, optimizer,
        vocabularies,
        device_descritpor,
    )

    if checkpoint is not None and not reset_step:
        trainer.step = checkpoint['step']

    return trainer
