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

import ynmt.hocon.arguments as harg
import ynmt.utilities.logging as logging


from ynmt.utilities.random import fix_random_procedure


def train(train_args):
    pass


def main():
    args = harg.get_arguments()
    fix_random_procedure(args.seed)
    train_args = harg.get_partial_arguments(args, 'binaries.train')
    train(train_args)


if __name__ == '__main__':
    main()
