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

from ynmt.models import register_model, Model

from ynmt.modules.encoders import TransformerEncoder
from ynmt.modules.decoders import TransformerDecoder
from ynmt.modules.perceptrons import MultilayerPerceptron

from ynmt.utilities.extractor import get_padding_mask, get_future_mask


@register_model('transformer')
class Transformer(Model):
    def __init__(self, args, dimension, encoder, decoder, generator):
        super(Transformer, self).__init__(args)
        assert dimension == encoder.dimension
        assert dimension == decoder.dimension
        self.dimension = dimension
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, source, target):
        source_mask = self.get_source_mask(source)
        target_mask = self.get_target_mask(target)

        codes = self.encoder(source, source_mask)

        hidden, cross_attention_weight = self.decoder(target, codes, target_mask, source_mask)

        prediction = self.generator(hidden)

        return prediction, cross_attention_weight

    def get_source_mask(self, source):
        source_pad_index = self.encoder.embed_token.padding_idx
        source_mask = get_padding_mask(source, source_pad_index).unsqueeze(1)
        return source_mask

    def get_target_mask(self, target):
        target_pad_index = self.decoder.embed_token.padding_idx
        target_mask = get_padding_mask(target, target_pad_index).unsqueeze(1)
        target_mask = target_mask | get_future_mask(target).unsqueeze(0)
        return target_mask

    @classmethod
    def setup(cls, args, task):
        transformer_encoder = TransformerEncoder(
            task.vocabularies['source'],
            args.encoder.layer_number,
            args.encoder.dimension,
            args.encoder.feedforward_dimension,
            args.encoder.head_number,
            args.encoder.dropout_probability,
            args.encoder.attention_dropout_probability,
            args.encoder.feedforward_dropout_probability,
            args.encoder.normalize_position
        )
        transformer_decoder = TransformerDecoder(
            task.vocabularies['target'],
            args.decoder.layer_number,
            args.decoder.dimension,
            args.decoder.feedforward_dimension,
            args.decoder.head_number,
            args.decoder.dropout_probability,
            args.decoder.attention_dropout_probability,
            args.decoder.feedforward_dropout_probability,
            args.decoder.normalize_position
        )

        generator = MultilayerPerceptron(args.decoder.dimension, len(task.vocabularies['target']), has_bias=False)
        torch.nn.init.normal_(generator.linear_layers[0].weight, mean=0, std=args.decoder.dimension ** -0.5)

        if args.share_enc_dec_embeddings:
            transformer_decoder.embed_token.weight = transformer_encoder.embed_token.weight

        if args.share_dec_io_embeddings:
            generator.linear_layers[0].weight = transformer_decoder.embed_token.weight

        transformer = cls(args, args.dimension, transformer_encoder, transformer_decoder, generator)

        return transformer
