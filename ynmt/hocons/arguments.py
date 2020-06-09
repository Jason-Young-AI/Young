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


from ynmt.pedestal.constant import Constant


class HOCONArguments(object):
    def __init__(self, hocon):
        if isinstance(hocon, pyhocon.config_tree.ConfigTree):
            self.__names = []
            for name in hocon:
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
                if name in self.__names:
                    if isinstance(hocon[name], pyhocon.config_tree.ConfigTree):
                        getattr(self, name).update(hocon[name])
                    else:
                        setattr(self, name, hocon[name])
                else:
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


def get_default_arguments(name):
    default_arguments = HOCONArguments(constant.DEFAULT_HOCON)
    name_list = name.split('.')
    arguments = default_arguments
    for name in name_list:
        arguments = getattr(arguments, name)
    return arguments


def get_user_arguments(name, user_hocon=None):
    default_arguments = get_default_arguments(name)
    if user_hocon is not None:
        user_arguments = default_arguments.update(user_hocon)
    else:
        user_arguments = default_arguments
    return user_arguments


def get_command_line_argument_parser():
    argument_parser = argparse.ArgumentParser(allow_abbrev=False)
    argument_parser.add_argument(
        '--config-filename',
        metavar='NAME',
        default='user.hocon',
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
