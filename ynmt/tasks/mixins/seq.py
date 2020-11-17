#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import collections

from yoolkit.cio import load_plain
from yoolkit.multiprocessing import multi_process


class SeqMixin(object):

    def count_tokens_of_corpus_partition(self, corpus_partition):
        corpus_partition_token_counter = collections.Counter()
        for line in corpus_partition:
            tokens = line.strip().split()
            corpus_partition_token_counter.update(tokens)
        return corpus_partition_token_counter

    def count_tokens_of_corpus(self, corpus_path, number_worker, work_amount):
        corpus_token_counter = collections.Counter()
        corpus_partitions = load_plain(corpus_path, partition_unit='line', partition_size=work_amount)
        corpus_partition_token_counters = multi_process(
            self.count_tokens_of_corpus_partition,
            corpus_partitions,
            number_worker
        )
        for corpus_partition_token_counter in corpus_partition_token_counters:
            corpus_token_counter.update(corpus_partition_token_counter)
        return corpus_token_counter
