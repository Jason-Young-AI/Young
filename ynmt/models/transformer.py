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


def build_model_transformer(args, vocabularies, checkpoint):
    transformer_encoder = TransformerEncoder(
        vocabularies['source'],
        args.encoder.layer_number, args.dimension, args.feedforward_dimension, args.head_number,
        args.dropout_probability, args.attention_dropout_probability, args.feedforward_dropout_probability
    )
    transformer_decoder = TransformerDecoder(
        vocabularies['target'],
        args.decoder.layer_number, args.dimension, args.feedforward_dimension, args.head_number,
        args.dropout_probability, args.attention_dropout_probability, args.feedforward_dropout_probability
    )

    transformer = Transformer(transformer_encoder, transformer_decoder)

    if checkpoint is not None:
        transformer.load_state_dict(checkpoint['model'], strict=False)

    return transformer


class Transformer(torch.nn.module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, source_length, target_length):
        codes = self.encoder(source, source_length)

        prediction, _, _ = self.decoder(codes, source_length, target[:-1], target_length)

        return prediction
