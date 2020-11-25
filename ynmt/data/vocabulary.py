#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from yoolkit.constant import Constant


constant = Constant()
constant.UNK = '<__UNK__>'
constant.PAD = '<__PAD__>'
constant.BOS = '<__BOS__>'
constant.EOS = '<__EOS__>'

constant.NUL = '<__NUL__>'
constant.LEN = '<__LEN__>'

constant.MSK = '<__MSK__>'
constant.SEP = '<__SEP__>'
constant.CLS = '<__CLS__>'


class Vocabulary(object):

    RESERVED_TOKENS = [
        constant.UNK,
        constant.PAD, constant.BOS, constant.EOS,
        constant.NUL,
        constant.LEN,
        constant.MSK, constant.SEP, constant.CLS,
    ]

    def __init__(self, tokens_and_frequencies, vocabulary_size=None, special_tokens=None):
        self.reserved_tokens = [ reserved_token for reserved_token in Vocabulary.RESERVED_TOKENS]

        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token in set(Vocabulary.RESERVED_TOKENS):
                    continue
                else:
                    self.reserved_tokens.append(special_token)

        for token, frequency in tokens_and_frequencies:
            if token in set(self.reserved_tokens):
                tokens_and_frequencies.remove((token, frequency))

        tokens_and_frequencies = sorted(tokens_and_frequencies, key=lambda x: x[0], reverse=False)
        tokens_and_frequencies = sorted(tokens_and_frequencies, key=lambda x: x[1], reverse=True)

        self.tokens_and_frequencies = tokens_and_frequencies[:vocabulary_size]

        self.index_to_token = dict()
        self.token_to_index = dict()

        index = 0
        for reserved_token in self.reserved_tokens:
            self.index_to_token[index] = reserved_token
            self.token_to_index[reserved_token] = index
            index += 1

        for token, frequency in self.tokens_and_frequencies:
            self.index_to_token[index] = token
            self.token_to_index[token] = index
            index += 1

        assert len(self.index_to_token) == len(self.token_to_index)
        self.length = index

    def __len__(self):
        return self.length

    def __iter__(self):
        for token, frequency in self.tokens_and_frequencies:
            yield token, frequency

    def __contains__(self, token):
        return token in self.tokens_and_frequencies

    def token(self, index):
        if index in self.index_to_token:
            token = self.index_to_token[index]
        else:
            token = self.unk_token
        return token

    def index(self, token):
        if token in self.token_to_index:
            index = self.token_to_index[token]
        else:
            index = self.unk_index
        return index

    @property
    def unk_token(self):
        return constant.UNK

    @property
    def unk_index(self):
        return self.index(constant.UNK)

    @property
    def pad_token(self):
        return constant.PAD

    @property
    def pad_index(self):
        return self.index(constant.PAD)

    @property
    def bos_token(self):
        return constant.BOS

    @property
    def bos_index(self):
        return self.index(constant.BOS)

    @property
    def eos_token(self):
        return constant.EOS

    @property
    def eos_index(self):
        return self.index(constant.EOS)

    @property
    def nul_token(self):
        return constant.NUL

    @property
    def nul_index(self):
        return self.index(constant.NUL)

    @property
    def len_token(self):
        return constant.LEN

    @property
    def len_index(self):
        return self.index(constant.LEN)

    @property
    def msk_token(self):
        return constant.MSK

    @property
    def msk_index(self):
        return self.index(constant.MSK)

    @property
    def sep_token(self):
        return constant.SEP

    @property
    def sep_index(self):
        return self.index(constant.SEP)

    @property
    def cls_token(self):
        return constant.CLS

    @property
    def cls_index(self):
        return self.index(constant.CLS)
