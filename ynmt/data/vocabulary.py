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


class Vocabulary(object):

    UNK_TOKEN = constant.UNK
    PAD_TOKEN = constant.PAD
    BOS_TOKEN = constant.BOS
    EOS_TOKEN = constant.EOS
    NUL_TOKEN = constant.NUL

    RESERVED_TOKENS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, NUL_TOKEN]

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

        self.tokens_and_frequencies = tokens_and_frequencies
        self.index_to_token = list()
        self.token_to_index = dict()

        for reserved_token in self.reserved_tokens:
            self.index_to_token.append(reserved_token)

        for token, frequency in self.tokens_and_frequencies[:vocabulary_size]:
            self.index_to_token.append(token)

        for index, token in enumerate(self.index_to_token):
            self.token_to_index[token] = index

    def __len__(self):
        return len(self.index_to_token)

    def token(self, index):
        if index < 0 or len(self) < index:
            return self.index_to_token[self.unk_index]
        else:
            return self.index_to_token[index]

    def index(self, token):
        if token not in self.token_to_index:
            return self.token_to_index[self.unk_token]
        else:
            return self.token_to_index[token]

    @property
    def unk_token(self):
        return Vocabulary.UNK_TOKEN

    @property
    def pad_token(self):
        return Vocabulary.PAD_TOKEN

    @property
    def bos_token(self):
        return Vocabulary.BOS_TOKEN

    @property
    def eos_token(self):
        return Vocabulary.EOS_TOKEN

    @property
    def nul_token(self):
        return Vocabulary.NUL_TOKEN

    @property
    def unk_index(self):
        return self.token_to_index[Vocabulary.UNK_TOKEN]

    @property
    def pad_index(self):
        return self.token_to_index[Vocabulary.PAD_TOKEN]

    @property
    def bos_index(self):
        return self.token_to_index[Vocabulary.BOS_TOKEN]

    @property
    def eos_index(self):
        return self.token_to_index[Vocabulary.EOS_TOKEN]

    @property
    def nul_index(self):
        return self.token_to_index[Vocabulary.NUL_TOKEN]
