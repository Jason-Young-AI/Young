#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-04-02 08:23
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import ynmt.hocons.arguments as harg


def preprocess(args):
    pass


def main():
    command_line_argument_parser = harg.get_command_line_argument_parser()
    command_line_arguments = command_line_argument_parser.parse_args()

    config_filename = command_line_arguments.config_filename
    config_filetype = command_line_arguments.config_filetype
    config_load_dir = command_line_arguments.config_load_dir
    assert os.path.isdir(config_load_dir), 'The configuration load directory does not exist!'
    config_save_dir = command_line_arguments.config_save_dir
    assert os.path.isdir(config_save_dir), 'The configuration save directory does not exist!'

    config_load_filepath = os.path.join(config_load_dir, config_filename)
    config_save_filepath = os.path.join(config_save_dir, config_filename)

    args = harg.get_default_arguments('binaries.preprocess')
    hocon = harg.load_hocon(config_load_filepath)
    args.update(hocon, 'binaries.preprocess')
    args.save(config_save_filepath)

    preprocess(args)


if __name__ == '__main__':
    main()
