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

import cupy

def main():
    parser = options.get_generation_parser(default_task='document_translation')
    youngseq_options.add_document_generation_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    main()
