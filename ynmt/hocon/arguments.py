#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-04-01 20:09
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import argparse
import collections

from yoolkit.constant import Constant
from yoolkit.arguments import load_arguments, update_arguments


constant = Constant()
constant.DEFAULT_ARGUMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
constant.DEFAULT_ARGUMENTS_PATH = os.path.join(constant.DEFAULT_ARGUMENTS_DIR, 'default.hocon')
constant.DEFAULT_ARGUMENTS = load_arguments(constant.DEFAULT_ARGUMENTS_PATH)


def get_command_line_argument_parser():
    argument_parser = argparse.ArgumentParser(allow_abbrev=True, formatter_class=argparse.RawTextHelpFormatter)
    argument_parser.add_argument(
        '-l',
        '--config-load-path',
        metavar='PATH',
        default='',
        help=('The path of configuration file to be loaded,\n'
              ' default configuration will be loaded if set to \'\'.\n'
              'DEFAULT=\'\''),
    )
    argument_parser.add_argument(
        '-s',
        '--config-save-path',
        metavar='PATH',
        default='',
        help=('The path of configuration file to be saved,\n'
              'configuration file will not be saved if set to \'\'.\n'
              'DEFAULT=\'\''),
    )
    argument_parser.add_argument(
        '-t',
        '--config-type',
        metavar='TYPE',
        default='hocon',
        choices=['hocon', 'json', 'yaml', 'properties'],
        help=('The type of configuration file to be loaded/saved.\n'
              'Choices=[\'hocon\', \'json\', \'yaml\', \'properties\']\n'
              'DEFAULT=\'hocon\''),
    )
    return argument_parser


def get_arguments():
    command_line_argument_parser = get_command_line_argument_parser()
    command_line_arguments = command_line_argument_parser.parse_args()

    config_load_path = command_line_arguments.config_load_path
    config_save_path = command_line_arguments.config_save_path
    config_type = command_line_arguments.config_type
    if os.path.isfile(config_load_path):
        print('User configuration found -> Using user configuration.')
        user_arguments = update_arguments(constant.DEFAULT_ARGUMENTS, config_load_path)
    else:
        print('No user configuration found -> Using default configuration.')
        user_arguments = constant.DEFAULT_ARGUMENTS

    if config_save_path == '':
        print('Configuration saving path not specified ! Saving no configuration.')
    else:
        user_arguments.save(config_save_path, output_type=config_type)
        print(f'Saving user configuration file: {config_save_path}, type={config_type}')

    if not os.path.isfile(config_load_path) and config_save_path != '':
        sys.exit(0)

    return user_arguments


def get_partial_arguments(arguments, name=None):
    partial_arguments = arguments
    if name is not None:
        name_list = name.split('.')
        for sub_name in name_list:
            partial_arguments = getattr(partial_arguments, sub_name)

    return partial_arguments
