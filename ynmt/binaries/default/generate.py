#!/usr/bin/env python3
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
import importlib
import pickle


import ynmt.hocon.arguments as harg


from ynmt.data.instance import InstanceSizeCalculator
from ynmt.data.iterator import RawTextIterator
from ynmt.utilities.line import tokenize, numericalize
from ynmt.utilities.logging import setup_logger, logging_level
from ynmt.utilities.checkpoint import load_checkpoint
from ynmt.utilities.distributed import get_device_descriptor

from ynmt.models import build_model
from ynmt.generators import build_generator


def generate(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])

    if args.device == 'CPU':
        logger.info(' * Generate on CPU ...')
    if args.device == 'GPU':
        logger.info(' * Generate on GPU ...')
        assert torch.cuda.device_count() > 0, f'Insufficient GPU!'

    device_descriptor = get_device_descriptor(args.device, 0)

    assert os.path.isfile(args.data.checkpoint_path), f'Checkpoint {args.data.checkpoint_path} does not exist!'
    checkpoint = load_checkpoint(args.data.checkpoint_path)
    logger.info(f' * Loaded checkpoint from \'{args.data.checkpoint_path}\'')

    vocabularies = checkpoint['vocabularies']

    sides = dict(args.data.sides)
    input_paths = dict(args.data.input_paths)
    output_paths = dict(args.data.output_paths)

    instance_handlers = dict()
    for side_name in sides.keys():
        instance_handlers[side_name] = lambda string: numericalize(tokenize(string), vocabularies[side_name])

    batches = RawTextIterator(
        input_paths,
        instance_handlers,
        args.data.batch_size,
        InstanceSizeCalculator(set(sides.keys()), args.data.batch_type)
    )

    # Build Model
    logger.info(f' * Building Model ...')
    model = build_model(checkpoint['model_settings'], vocabularies)
    logger.info(f'   Loading Parameters ...')
    model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f'   Parameters Loaded.')

    # Build Generator
    logger.info(f' * Building Generator ...')
    generator = build_generator(
        args.generator,
        model,
        vocabularies,
        output_paths,
        device_descriptor,
        logger
    )
    logger.info(f'   Generator \'{args.generator.name}\' built.')

    logger.info(f' * Generating ...')
    generator.launch(batches)
    logger.info(' $ Finished !')


def main():
    args = harg.get_arguments()
    generate_args = harg.get_partial_arguments(args, 'binaries.generate')
    generate_args.generator = harg.get_partial_arguments(args, f'generators.{generate_args.generator}')
    generate(generate_args)


if __name__ == '__main__':
    main()
