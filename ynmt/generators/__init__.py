#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-08-12 18:31
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ynmt.generators.generator import Generator

from ynmt.generators.seq2seq import build_generator_seq2seq


def build_generator(args,
                    model,
                    vocabularies,
                    device_descriptor,
                    logger):

    logger.info(f' | Moving model to device of Generator ...')
    model.to(device_descriptor)
    logger.info(f' - Completed.')

    generator = globals()[f'build_generator_{args.name}'](
        args,
        model,
        vocabularies,
        device_descriptor,
        logger
    )

    return generator
