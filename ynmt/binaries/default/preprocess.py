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


from yoolkit.logging import setup_logger, logging_level

import ynmt.hocon.arguments as harg

from ynmt.tasks import build_task

from ynmt.utilities.random import fix_random_procedure


def preprocess(args):
    logger = setup_logger(name=args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])

    logger.disabled = args.logger.off

    fix_random_procedure(args.random_seed)

    logger.info(f'Building Task: \'{args.task.name}\' ...')
    task = build_task(args.task, logger)
    logger.info(f'The construction of Task is complete.')

    logger.info(f'Building Ancillary Datasets ...')
    task.build_ancillary_datasets(args.task)
    logger.info(f'The construction of Ancillary Datasets is complete.')

    logger.info(f'Building Datasets ...')
    task.build_datasets(args.task)
    logger.info(f'The construction of Datasets is complete.')

    logger.info(f' $ Finished !')


def main():
    args = harg.get_arguments()
    preprocess_args = harg.get_partial_arguments(args, 'binaries.preprocess')
    preprocess_args.task = harg.get_partial_arguments(args, f'tasks.{preprocess_args.task}')
    preprocess(preprocess_args)


if __name__ == '__main__':
    main()
