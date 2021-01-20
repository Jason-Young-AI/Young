#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:03
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
from ynmt.hocon.arguments import constant, load_arguments


def get_command_line_argument_parser():
    argument_parser = argparse.ArgumentParser(allow_abbrev=True, formatter_class=argparse.RawTextHelpFormatter)
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
        help=('The type of configuration file to be saved.\n'
              'Choices=[\'hocon\', \'json\', \'yaml\', \'properties\']\n'
              'DEFAULT=\'hocon\''),
    )
    return argument_parser


def main():
    command_line_argument_parser = get_command_line_argument_parser()
    command_line_arguments = command_line_argument_parser.parse_args()

    config_save_path = command_line_arguments.config_save_path
    config_type = command_line_arguments.config_type

    if config_save_path != '':
        arguments = load_arguments(constant.ALL_SYSTEM_COMPONENTS_ARGUMENTS_PATH)
        arguments.save(config_save_path, output_type=config_type)
        print(f'Saving configuration file: {config_save_path}, type={config_type}')
    else:
        print(f'                >   Welcome to use YoungNMT!   <                ')
        print(f'----------------------------------------------------------------')
        print()
        print(f'Please use the following command to make the most of the system:')
        print(f'0. ynmt --help')
        print(f'1. ynmt-preprocess --help')
        print(f'2. ynmt-train --help')
        print(f'3. ynmt-test --help')
        print(f'4. ynmt-serve --help')
        print()

    sys.stdout.flush()


if __name__ == '__main__':
    main()
