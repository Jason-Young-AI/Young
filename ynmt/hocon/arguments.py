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
import pyhocon
import argparse
import collections


from ynmt.utilities.constant import Constant


class HOCONArguments(object):
    RESERVED_NAME = set({ '_HOCONArguments__names', 'dictionary', 'hocon', 'update', 'save' })
    def __init__(self, hocon):
        if isinstance(hocon, pyhocon.config_tree.ConfigTree):
            self.__names = []
            for name in hocon:
                if name in HOCONArguments.RESERVED_NAME:
                    raise ValueError(f'Invalid hocon attribute name(\'{name}\')')
                self.__names.append(name)
                if isinstance(hocon[name], pyhocon.config_tree.ConfigTree):
                    setattr(self, name, HOCONArguments(hocon[name]))
                else:
                    setattr(self, name, hocon[name])
        else:
            raise ValueError(f'Argument hocon(\'{hocon}\') is not a \'pyhocon.config_tree.ConfigTree\' object.')

    @property
    def dictionary(self):
        dictionary = {}
        for name in self.__names:
            if isinstance(getattr(self, name), HOCONArguments):
                dictionary[name] = getattr(self, name).dictionary
            else:
                dictionary[name] = getattr(self, name)
        dictionary = collections.OrderedDict(sorted(dictionary.items(), key=lambda x: x[0]))
        return dictionary

    @property
    def hocon(self):
        hocon = pyhocon.ConfigFactory().from_dict(self.dictionary)
        return hocon

    def update(self, hocon):
        if isinstance(hocon, pyhocon.config_tree.ConfigTree):
            for name in hocon:
                if name in HOCONArguments.RESERVED_NAME:
                    raise ValueError(f'Invalid hocon attribute name(\'{name}\')')
                if name in self.__names:
                    if isinstance(hocon[name], pyhocon.config_tree.ConfigTree):
                        getattr(self, name).update(hocon[name])
                    else:
                        setattr(self, name, hocon[name])
                else:
                    self.__names.append(name)
                    setattr(self, name, HOCONArguments(hocon[name]))
        else:
            raise ValueError(f'Argument hocon(\'{hocon}\') is not a \'pyhocon.config_tree.ConfigTree\' object.')

    def save(self, output_path, output_type='hocon'):
        output_string = pyhocon.converter.HOCONConverter.convert(self.hocon, output_type)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(output_string)


constant = Constant()
constant.DEFAULT_HOCON_DIR = os.path.abspath(os.path.dirname(__file__))
constant.DEFAULT_HOCON_PATH = os.path.join(constant.DEFAULT_HOCON_DIR, 'default.hocon')
constant.DEFAULT_HOCON = pyhocon.ConfigFactory.parse_file(constant.DEFAULT_HOCON_PATH)


def load_hocon(hocon_abs_path):
    return pyhocon.ConfigFactory.parse_file(hocon_abs_path)


def get_default_arguments():
    default_arguments = HOCONArguments(constant.DEFAULT_HOCON)
    return default_arguments


def get_user_arguments(user_hocon=None):
    default_arguments = get_default_arguments()
    if user_hocon is not None:
        default_arguments.update(user_hocon)
    user_arguments = default_arguments
    return user_arguments


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
        user_hocon = load_hocon(config_load_path)
        user_arguments = get_user_arguments(user_hocon)
    else:
        print('No user configuration found -> Using default configuration.')
        user_arguments = get_user_arguments()

    if config_save_path == '':
        print('Configuration saving path not specified ! Saving no configuration.')
    else:
        user_arguments.save(config_save_path, output_type=config_type)
        print(f'Saving user configuration file: {config_save_path}, type={config_type}')

    return user_arguments


def get_partial_arguments(arguments, name=None):
    partial_arguments = arguments
    if name is not None:
        name_list = name.split('.')
        for sub_name in name_list:
            partial_arguments = getattr(partial_arguments, sub_name)

    return partial_arguments
