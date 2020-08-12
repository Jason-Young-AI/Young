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

from ynmt.schedulers import build_scheduler
from ynmt.optimizers import build_optimizer

from ynmt.trainers.seq2seq import build_trainer_seq2seq

from ynmt.utilities.checkpoint import load_checkpoint


def build_trainer(args,
                  model,
                  vocabularies,
                  device_descriptor,
                  logger, visualizer):

    logger.info(f' | Moving model to device of Trainer ...')
    model.to(device_descriptor)
    logger.info(f' - Completed.')

    logger.info(f' | Checking already exist checkpoint ...')
    checkpoint = load_checkpoint(args.checkpoint.directory, args.checkpoint.name)
    if checkpoint is None:
        logger.info(f' - No checkpoint found in \'{args.checkpoint.directory}\'.')
    else:
        logger.info(f' - Loaded latest checkpoint from \'{args.checkpoint.directory}\' at {checkpoint["step"]} steps')


    # Build Scheduler
    logger.info(f' | Building Learning Rate Scheduler ...')
    scheduler = build_scheduler(args.scheduler, model)
    logger.info(f' - Scheduler \'{args.scheduler.name}\' built.')

    # Build Optimizer
    logger.info(f' | Building Optimizer ...')
    optimizer = build_optimizer(args.optimizer, model)
    logger.info(f' - Optimizer \'{args.optimizer.name}\' built.')

    if checkpoint is not None:
        if not args.checkpoint.reset_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f' | Reset Scheduler.')

        if not args.checkpoint.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f' | Reset Optimizer.')

        logger.info(f' | Loading Parameters ...')
        model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f' - Parameters Loaded.')

    trainer = globals()[f'build_trainer_{args.name}'](
        args,
        model,
        scheduler, optimizer,
        vocabularies,
        device_descriptor,
        logger, visualizer
    )

    if checkpoint is not None and not args.checkpoint.reset_step:
        trainer.step = checkpoint['step']

    return trainer
