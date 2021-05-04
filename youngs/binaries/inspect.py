#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2021-05-03 12:55
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

import youngs.hocon.arguments as harg

from youngs.utilities.checkpoint import load_checkpoint, save_checkpoint

from yoolkit.logging import setup_logger, logging_level


def average_models(args, logger):
    if len(args.checkpoint_paths) <= 1:
        logger.info(f'The number of checkpoint_paths must larger than one.\n')
        return

    logger.info(
        f'The models that loaded from checkpoints will be averaged: {args.checkpoint_paths}.\n'
        f'There are total {len(args.checkpoint_paths)} checkpoints will be loaded.'
    )
    checkpoint_number = len(args.checkpoint_paths)
    new_checkpoint = load_checkpoint(args.checkpoint_paths[0])

    assert isinstance(args.new_step, int), f'The type of new_step must be Int().'
    if args.new_step > 0:
        logger.info(f'Training step is reset to [\'{args.new_step}\'].')
        new_checkpoint['step'] = args.new_step
    else:
        logger.info(f'Training step is set to be same as the 1st checkpoint.')

    for index, checkpoint_path in enumerate(args.checkpoint_paths[1:]):
        n = index + 1
        checkpoint = load_checkpoint(checkpoint_path)
        if list(checkpoint['model_state'].keys()) != list(new_checkpoint['model_state'].keys()):
            raise KeyError(f'The keys of checkpoint ({checkpoint_path}) does not match the first!')

        for parameter_key, parameter_value in checkpoint['model_state'].items():
            new_checkpoint['model_state'][parameter_key].mul_(n).add_(parameter_value).div_(n+1)

    save_checkpoint(new_checkpoint, args.output_path)


def inspect(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report, clean_logging_file=True)

    logger.info(
        f'\n The following is the arguments:'
        f'\n{args}'
    )

    if args.option == 'average_models':
        logger.info('Averaging Models ...')
        average_models(args.average_models, logger)

    logger.info(' $ Finished !')


def main():
    inspect_args = harg.get_arguments('inspect')
    inspect(inspect_args)


if __name__ == '__main__':
    main()
