#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:08
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch

from ynmt.testers import register_tester, Tester
from ynmt.testers.ancillaries import GreedySearcher

from ynmt.data.batch import Batch
from ynmt.data.instance import Instance
from ynmt.data.iterator import RawTextIterator
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.metrics import BLEUScorer
from ynmt.utilities.sequence import stringize, numericalize, tokenize, dehyphenate
from ynmt.utilities.extractor import get_tiled_tensor


@register_tester('fast_wait_k')
class FastWaitK(Tester):
    def __init__(self,
        factory, model,
        greedy_searcher,
        bpe_symbol, remove_bpe, dehyphenate,
        reference_path,
        wait_source_time,
        output_directory, output_name,
        device_descriptor, logger
    ):
        super(FastWaitK, self).__init__(factory, model, output_directory, output_name, device_descriptor, logger)
        self.greedy_searcher = greedy_searcher

        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe
        self.dehyphenate = dehyphenate
        self.reference_path = reference_path

        self.wait_source_time = wait_source_time

    @classmethod
    def setup(cls, settings, factory, model, device_descriptor, logger):
        args = settings.args

        greedy_searcher = GreedySearcher(
            search_space_size = len(factory.vocabularies['target']),
            initial_node = factory.vocabularies['target'].bos_index,
            terminal_node = factory.vocabularies['target'].eos_index,
            min_depth = args.greedy_searcher.min_length, max_depth = args.greedy_searcher.max_length,
        )

        tester = cls(
            factory, model,
            greedy_searcher,
            args.bpe_symbol, args.remove_bpe, args.dehyphenate,
            args.reference_path,
            args.wait_source_time,
            args.outputs.directory, args.outputs.name,
            device_descriptor, logger
        )

        return tester

    def initialize(self, output_extension):
        output_basepath = self.output_basepath + '.' + output_extension

        self.total_sentence_number = 0

        self.trans_path = output_basepath + '.trans'
        with open(self.trans_path, 'w', encoding='utf-8') as trans_file:
            trans_file.truncate()

    def customize_batch(self, batch):
        padded_batch = Batch(set({'source', }))
        padded_source_attributes, _ = pad_attribute(batch.source, self.factory.vocabularies['source'].pad_index)
        padded_batch.source = torch.tensor(padded_source_attributes, dtype=torch.long, device=self.device_descriptor)
        return padded_batch

    def test(self, customized_batch):
        source = customized_batch.source
        parallel_line_number, max_source_length = source.size()

        self.greedy_searcher.initialize(parallel_line_number, self.device_descriptor)

        while not self.greedy_searcher.finished:
            temp_source = torch.index_select(source, 0, self.greedy_searcher.line_original_indices)
            source_mask = self.model.get_source_mask(temp_source)
            codes = self.model.encoder(temp_source, source_mask)

            previous_prediction = self.greedy_searcher.found_nodes
            previous_prediction_mask = self.model.get_target_mask(previous_prediction)
            cross_attention_weight_mask = self.model.get_cross_attention_weight_mask(previous_prediction, temp_source, self.wait_source_time + 1) # +1 for bos

            hidden, cross_attention_weight = self.model.decoder(
                previous_prediction,
                codes,
                previous_prediction_mask,
                cross_attention_weight_mask
            )

            logits = self.model.generator(hidden)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            prediction_distribution = log_probs[:, -1, :]
            self.greedy_searcher.search(prediction_distribution)
            self.greedy_searcher.update()

        return self.greedy_searcher.result

    def output(self, result):
        parallel_line_number = len(result)

        with open(self.trans_path, 'a', encoding='utf-8') as trans_file:
            for line_index in range(parallel_line_number):
                lprob = result[line_index]['log_prob']
                trans = result[line_index]['path']

                trans_tokens = stringize(trans, self.factory.vocabularies['target'])
                trans_sentence = ' '.join(trans_tokens)

                # Final Trans
                if self.remove_bpe:
                    trans_sentence = (trans_sentence + ' ').replace(self.bpe_symbol, '').strip()
                if self.dehyphenate:
                    trans_sentence = dehyphenate(trans_sentence)
                trans_file.writelines(trans_sentence + '\n')

    def report(self):
        bleu_scorer = BLEUScorer()
        bleu_scorer.initialize()
        with open(self.trans_path, 'r', encoding='utf-8') as trans_file, open(self.reference_path, 'r', encoding='utf-8') as reference_file:
            for trans_sentence, reference_sentence in zip(trans_file, reference_file):
                bleu_scorer.add(trans_sentence.lower().split(), [reference_sentence.lower().split(), ])

        bleu_score = bleu_scorer.result_string
        self.logger.info('   ' + bleu_score)
