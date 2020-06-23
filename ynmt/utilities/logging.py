#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-05-26 23:41
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import tempfile


def get_logger(logging_path='', logging_level=logging.NOTSET):
    logging_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    if logging_path == '':
        sys_temp_dir_path = tempfile.gettempdir()
        temp_dir_path = tempfile.mkdtemp(dir=sys_temp_dir_path, prefix='ynmt-')
        logging_file, logging_path = tempfile.mkstemp(dir=temp_dir_path)
        print(f'Logging path is not specified, the following path is used for logging: {logging_path}')

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(logging_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logging_path)
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)

    return logger