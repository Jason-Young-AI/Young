#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-12-13 16:39
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch

from yoolkit.logging import setup_logger, logging_level
from yoolkit.registration import import_modules

import youngs.hocon.arguments as harg

from youngs.utilities.checkpoint import load_checkpoint
from youngs.utilities.distributed import get_device_descriptor

from youngs.factories import build_factory
from youngs.models import build_model
from youngs.servers import build_server
from youngs.testers import build_tester


def serve_web(args):
    import_modules(args.user_defined_modules_directory, 'youngs.user_defined')
    logger = setup_logger(args.logger.name, logging_path=args.logger.path+'-web', logging_level=logging_level['INFO'], to_console=args.logger.console_report, clean_logging_file=True)

    logger.info(
        f'\n The following is the arguments:'
        f'\n{args}'
    )

    # Build Server
    logger.info(f' 1.Building Server ...')
    server = build_server(args.server, None, logger)
    logger.info(f'   The construction of Server [\'{args.server.name}\' : \'{server.__class__.__name__}\'] is complete.')

    # Launch Server
    logger.info(f' 2.Launch Server ...')
    server.launch('web')


def serve_app(args):
    import_modules(args.user_defined_modules_directory, 'youngs.user_defined')
    logger = setup_logger(args.logger.name, logging_path=args.logger.path+'-app', logging_level=logging_level['INFO'], to_console=args.logger.console_report, clean_logging_file=True)

    logger.info(
        f'\n The following is the arguments:'
        f'\n{args}'
    )

    if args.device == 'CPU':
        logger.info(' * Serve on CPU ...')
    if args.device == 'GPU':
        logger.info(' * Serve on GPU ...')
        assert torch.cuda.device_count() >= 1, f'Insufficient GPU!'
    device_descriptor = get_device_descriptor(args.device, 0)

    import_modules(args.user_defined_modules_directory, 'youngs.user_defined')

    # Find checkpoint
    assert os.path.isfile(args.checkpoint_path), f'Checkpoint \'{args.checkpoint_path}\' does not exist!'

    # Build Factory
    logger.info(f' + Building Factory ...')
    factory = build_factory(args.factory, logger)
    logger.info(f'   The construction of Factory [\'{args.factory.name}\' : \'{factory.__class__.__name__}\'] is complete.')

    # Load Ancillary Datasets
    logger.info(f' + Loading Ancillary Datasets ...')
    factory.load_ancillary_datasets(args.factory.args)
    logger.info(f'   Ancillary Datasets has been loaded.')

    # Load Checkpoint & Build Model
    checkpoint = load_checkpoint(args.checkpoint_path)
    logger.info(f'   Checkpoint has been loaded from [\'{args.checkpoint_path}\']')
    model_settings = checkpoint["model_settings"]
    logger.info(f' 1.Building Model ...')
    model = build_model(model_settings, factory)
    logger.info(f'   The construction of Model [\'{model_settings.name}\' : \'{model.__class__.__name__}\'] is complete.')

    logger.info(f' 2.Loading Parameters ...')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    logger.info(f'   Loaded.')

    logger.info(f' 3.Moving model to the specified device ...')
    model.to(device_descriptor)
    logger.info(f'   Complete.')

    # Build Tester
    logger.info(f' 4.Building Tester ...')
    tester = build_tester(args.tester, factory, model, device_descriptor, logger)
    logger.info(f'   The construction of Tester [\'{args.tester.name}\' : \'{tester.__class__.__name__}\'] is complete.')

    # Build Server
    logger.info(f' 5.Building Server ...')
    server = build_server(args.server, tester, logger)
    logger.info(f'   The construction of Server [\'{args.server.name}\' : \'{server.__class__.__name__}\'] is complete.')

    # Launch Server
    logger.info(f' 6.Launch Server ...')
    server.launch('app')


def serve(args):
    assert args.serve_type in set({'web', 'app'}), f'Wrong type of serve_type: [\'{serve_type}\']'
    if args.serve_type == 'web':
        serve_web(args)
    if args.serve_type == 'app':
        serve_app(args)


def main():
    serve_args = harg.get_arguments('serve')
    serve(serve_args)


if __name__ == '__main__':
    main()
