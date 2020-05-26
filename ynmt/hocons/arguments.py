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


class HOCONArguments(object):
    def __init__(self, hocon, name):
        if isinstance(hocon, pyhocon.config_tree.ConfigTree):
            if name in hocon:
                self.__name = name
                self.__succeed_names = []
                if isinstance(hocon[name], pyhocon.config_tree.ConfigTree):
                    self.__value = None
                    for succeed_name in hocon[name]:
                        self.__succeed_names.append(succeed_name)
                        setattr(self, succeed_name, HOCONArguments(hocon[name], succeed_name))
                else:
                    self.__value = hocon[name]
            else:
                raise ValueError(f'Hocon(\'{hocon}\') does not have name(\'{name}\').')
        else:
            raise ValueError(f'Argument hocon(\'{hocon}\') is not a \'pyhocon.config_tree.ConfigTree\' object.')

    def __setattr__(self, key, value):
        if key in ['_HOCONArguments__name', '_HOCONArguments__succeed_names']:
            if key in self.__dict__:
                raise AttributeError(f'Can not modify Reserved Attribute \'{key}\'.')
            else:
                self.__dict__[key] = value
        elif key in self.__succeed_names:
            if key in self.__dict__:
                raise AttributeError(f'Can not rewrite HOCONArguments object attribute \'{key}\'.')
            else:
                self.__dict__[key] = value
        elif key in ['_HOCONArguments__value',]:
            self.__dict__[key] = value
        else:
            raise AttributeError(f'This Attribute: \'{key}\' does not exist.')

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    def update_value(self, new_value):
        self.__value == new_value

    def update(self, hocon, name):
        if name == self.name:
            if isinstance(hocon, pyhocon.config_tree.ConfigTree):
                if name in hocon:
                    if isinstance(hocon[name], pyhocon.config_tree.ConfigTree):
                        for succeed_name in hocon[name]:
                            if succeed_name in self.__succeed_names:
                                succeed = getattr(self, succeed_name)
                                succeed.update(hocon[name], succeed_name)
                            else:
                                self.__succeed_names.append(succeed_name)
                                setattr(self, succeed_name, HOCONArguments(hocon[name], succeed_name))
                    else:
                        self.update_value(hocon[name])
                else:
                    raise ValueError(f'Hocon(\'{hocon}\') does not have name(\'{name}\').')
            else:
                raise ValueError(f'Argument hocon(\'{hocon}\') is not a \'pyhocon.config_tree.ConfigTree\' object.')
        else:
            raise ValueError(f'Argument name(\'{name}\') must match obj.name(\'{self.name}\').')

    def save(self, path):
        pass


DEFAULT_HOCON_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_HOCON_PATH = os.path.join(DEFAULT_HOCON_DIR, 'default.hocon')
DEFAULT_HOCON = pyhocon.ConfigFactory.parse_file(DEFAULT_HOCON_PATH)

def load_hocon(hocon_abs_path):
    return pyhocon.ConfigFactory.parse_file(hocon_abs_path)


def get_default_arguments(name):
    default_arguments = HOCONArguments(DEFAULT_HOCON, name)
    return default_arguments


def get_user_arguments(name, user_hocon=None):
    default_arguments = get_default_arguments(name)
    if user_hocon is not None:
        user_arguments = default_arguments.update(user_hocon, name)
    else:
        user_arguments = default_arguments
    return user_arguments


def get_command_line_argument_parser():
    argument_parser = argparse.ArgumentParser(allow_abbrev=False)
    argument_parser.add_argument(
        '--config-filename',
        metavar='NAME',
        default='',
        help='The name of configuration file to be loaded/saved',
    )
    argument_parser.add_argument(
        '--config-filetype',
        metavar='TYPE',
        default='hocon',
        choices=['hocon', 'json', 'yaml', 'properties'],
        help='The type of configuration file to be loaded/saved',
    )
    argument_parser.add_argument(
        '--config-load-dir',
        metavar='PATH',
        default='.',
        help='The dir of configuration file to be loaded',
    )
    argument_parser.add_argument(
        '--config-save-dir',
        metavar='PATH',
        default='.',
        help='The dir of configuration file to be saved',
    )
    return argument_parser
