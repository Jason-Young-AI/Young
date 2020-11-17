#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-04-02 08:23
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch

from yoolkit.logging import setup_logger, logging_level

import ynmt.hocon.arguments as harg

from ynmt.utilities.checkpoint import load_checkpoint
from ynmt.utilities.distributed import get_device_descriptor

from ynmt.tasks import build_task
from ynmt.models import build_model
from ynmt.testers import build_tester


def test(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)

    if args.device == 'CPU':
        logger.info(' * Test on CPU ...')
    if args.device == 'GPU':
        logger.info(' * Test on GPU ...')
        assert torch.cuda.device_count() > 0, f'Insufficient GPU!'

    device_descriptor = get_device_descriptor(args.device, 0)

    logger.info(f' * Building Task: \'{args.task.name}\' ...')
    task = build_task(args.task, logger)
    logger.info(f'   The construction of Task is complete.')

    # Load Ancillary Datasets
    logger.info(f' * Loading Ancillary Datasets ...')
    task.load_ancillary_datasets(args.task.args)
    logger.info(f'   Ancillary Datasets has been loaded.')

    # Build Tester
    logger.info(f' * Building Tester \'{args.tester.name}\' ...')
    tester = build_tester(args.tester, task, device_descriptor, logger)
    logger.info(f'   The construction of tester is completed.')

    assert os.path.isdir(args.checkpoint_directory), f' Checkpoint directory \'{args.checkpoint_directory}\' does not exist!'
    assert os.path.isdir(args.output_directory), f' Output directory \'{args.output_directory}\' does not exist!'

    listed_names = os.listdir(args.checkpoint_directory)
    checkpoint_names = list()
    checkpoint_paths = list()
    for listed_name in listed_names:
        listed_path = os.path.join(args.checkpoint_directory, listed_name)
        if os.path.isfile(listed_path):
            checkpoint_names.append(listed_name)
            checkpoint_paths.append(listed_path)

    logger.info(f'   There are {len(checkpoint_names)} checkpoints will be loaded: {checkpoint_names}')
    for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
        checkpoint = load_checkpoint(checkpoint_path)
        logger.info(f' * Checkpoint has been loaded from \'{checkpoint_path}\'')

        # Build Model
        model_settings = checkpoint["model_settings"]
        logger.info(f' * Building Model \'{model_settings.name}\' ...')
        model = build_model(model_settings, task)
        logger.info(f'   Loading Parameters ...')
        model.load_state_dict(checkpoint['model_state'], strict=False)
        logger.info(f'   Loaded.')

        logger.info(f' + Moving model to device of Tester ...')
        model.to(device_descriptor)
        logger.info(f' - Completed.')

        checkpoint_basename = os.path.splitext(checkpoint_name)[0]

        output_basepath = os.path.join(args.output_directory, checkpoint_basename)

        # Launch Tester
        logger.info(f' * Launch Tester ...')
        logger.info(f'   The testing outputs of the model will be wrote into \'{output_basepath}-*\'.')
        tester.initialize()
        tester.launch(model, output_basepath)

    logger.info(f' $ Finished !')


def main():
    test_args = harg.get_arguments('test')
    test(test_args)


if __name__ == '__main__':
    main()
