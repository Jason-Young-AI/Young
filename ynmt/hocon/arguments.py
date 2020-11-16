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
constant.ARGUMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
constant.ALL_SYSTEM_COMPONENTS_ARGUMENTS_PATH = os.path.join(constant.ARGUMENTS_DIR, '__all__.hocon')

constant.BINARIES_ARGUMENTS_DIR = os.path.join(constant.ARGUMENTS_DIR, 'binaries')

def get_binary_names(phase_name):
    assert phase_name in set({'preprocess', 'train', 'test'}), f'Wrong choice of process phase: {phase_name}!'

    binary_names = list()
    phase_dir = os.path.join(constant.BINARIES_ARGUMENTS_DIR, phase_name)
    item_names = os.listdir(phase_dir)
    for item_name in item_names:
        item_path = os.path.join(phase_dir, item_name)
        if os.path.isfile(item_path):
            if item_name.startswith('_') or item_name.startswith('.'):
                continue
            if item_name.endswith('.hocon'):
                binary_name = item_name[:item_name.find('.hocon')]
                binary_names.append(binary_name)

    return binary_names


def get_command_line_argument_parser(phase_name):
    argument_parser = argparse.ArgumentParser(allow_abbrev=True, formatter_class=argparse.RawTextHelpFormatter)

    binary_names = get_binary_names(phase_name)

    argument_parser.add_argument(
        '-n',
        '--name',
        metavar='NAME',
        default='seq2seq',
        choices=binary_names,
        help=('The name of configuration,\n'
              f'Choices={binary_names}\n'
              'DEFAULT=\'seq2seq\''),
    )

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


def get_arguments(phase_name):
    command_line_argument_parser = get_command_line_argument_parser(phase_name)
    command_line_arguments = command_line_argument_parser.parse_args()

    binary_name = command_line_arguments.name
    binary_path = os.path.join(constant.BINARIES_ARGUMENTS_DIR, phase_name, binary_name) + '.hocon'

    binary_arguments = load_arguments(binary_path)

    config_load_path = command_line_arguments.config_load_path
    config_save_path = command_line_arguments.config_save_path
    config_type = command_line_arguments.config_type

    if os.path.isfile(config_load_path):
        print('User configuration found -> Using user configuration.')
        user_arguments = update_arguments(binary_arguments, config_load_path)
    else:
        print('No user configuration found -> Using default configuration.')
        user_arguments = binary_arguments

    if config_save_path == '':
        print('Configuration saving path not specified ! Saving no configuration.')
    else:
        user_arguments.save(config_save_path, output_type=config_type)
        print(f'Saving user configuration file: {config_save_path}, type={config_type}')

    if not os.path.isfile(config_load_path) and config_save_path != '':
        sys.exit(0)

    return user_arguments
