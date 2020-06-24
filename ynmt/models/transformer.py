#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch


from ynmt.modules.encoders import TransformerEncoder
from ynmt.modules.decoders import TransformerDecoder


def build_transformer(args, vocabularies, checkpoint):
    encoder = 


class Transformer(torch.nn.module):
    def __init__(self, ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder
        self.decoder = TransformerDecoder


    def forward(self, source, target):
        pass
