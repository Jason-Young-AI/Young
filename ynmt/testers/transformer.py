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
from ynmt.testers.ancillaries import BeamSearcher

from ynmt.data.batch import Batch
from ynmt.data.instance import Instance
from ynmt.data.iterator import RawTextIterator
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.metrics import BLEUScorer
from ynmt.utilities.sequence import stringize, numericalize, tokenize, dehyphenate
from ynmt.utilities.extractor import get_tiled_tensor


@register_tester('transformer')
class Transformer(Tester):
    def __init__(self,
        factory, model,
        beam_searcher,
        bpe_symbol, remove_bpe, dehyphenate,
        reference_path,
        output_directory, output_name,
        device_descriptor, logger
    ):
        super(Transformer, self).__init__(factory, model, output_directory, output_name, device_descriptor, logger)
        self.beam_searcher = beam_searcher

        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe
        self.dehyphenate = dehyphenate
        self.reference_path = reference_path

    @classmethod
    def setup(cls, settings, factory, model, device_descriptor, logger):
        args = settings.args

        beam_searcher = BeamSearcher(
            reserved_path_number = args.beam_searcher.beam_size,
            candidate_path_number = args.beam_searcher.n_best,
            search_space_size = len(factory.vocabularies['target']),
            initial_node = factory.vocabularies['target'].bos_index,
            terminal_node = factory.vocabularies['target'].eos_index,
            min_depth = args.beam_searcher.min_length, max_depth = args.beam_searcher.max_length,
            alpha = args.beam_searcher.penalty.alpha, beta = args.beam_searcher.penalty.beta
        )

        tester = cls(
            factory, model,
            beam_searcher,
            args.bpe_symbol, args.remove_bpe, args.dehyphenate,
            args.reference_path,
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

        self.detailed_trans_path = output_basepath + '.detailed_trans'
        with open(self.detailed_trans_path, 'w', encoding='utf-8') as detailed_trans_file:
            detailed_trans_file.truncate()

    def customize_batch(self, batch):
        padded_batch = Batch(set({'source', }))
        padded_source_attributes, _ = pad_attribute(batch.source, self.factory.vocabularies['source'].pad_index)
        padded_batch.source = torch.tensor(padded_source_attributes, dtype=torch.long, device=self.device_descriptor)
        return padded_batch

    def test(self, customized_batch):
        source = customized_batch.source
        parallel_line_number, max_source_length = source.size()

        source_mask = self.model.get_source_mask(source)
        codes = self.model.encoder(source, source_mask)

        source_mask = get_tiled_tensor(source_mask, 0, self.beam_searcher.reserved_path_number)
        codes = get_tiled_tensor(codes, 0, self.beam_searcher.reserved_path_number)

        self.beam_searcher.initialize(parallel_line_number, self.device_descriptor)

        while not self.beam_searcher.finished:
            previous_prediction = self.beam_searcher.found_nodes.reshape(
                self.beam_searcher.parallel_line_number * self.beam_searcher.reserved_path_number,
                -1
            )
            previous_prediction_mask = self.model.get_target_mask(previous_prediction)

            hidden, cross_attention_weight = self.model.decoder(
                previous_prediction,
                codes,
                previous_prediction_mask,
                source_mask
            )

            logits = self.model.generator(hidden)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            prediction_distribution = log_probs[:, -1, :].reshape(
                self.beam_searcher.parallel_line_number,
                self.beam_searcher.reserved_path_number,
                -1
            )
            self.beam_searcher.search(prediction_distribution)
            self.beam_searcher.update()

            source_mask = source_mask.index_select(0, self.beam_searcher.path_offset.reshape(-1))
            codes = codes.index_select(0, self.beam_searcher.path_offset.reshape(-1))

        return self.beam_searcher.candidate_paths

    def output(self, result):
        candidate_paths = result
        parallel_line_number = len(candidate_paths)

        with open(self.trans_path, 'a', encoding='utf-8') as trans_file, open(self.detailed_trans_path, 'a', encoding='utf-8') as detailed_trans_file:
            for line_index in range(parallel_line_number):

                detailed_trans_file.writelines(f'No.{self.total_sentence_number}:\n')
                self.total_sentence_number += 1

                candidate_path_number = len(candidate_paths[line_index])
                for path_index in range(candidate_path_number):
                    candidate_path = candidate_paths[line_index][path_index]

                    lprob = candidate_path['log_prob']
                    score = candidate_path['score']
                    trans = candidate_path['path']

                    trans_tokens = stringize(trans, self.factory.vocabularies['target'])
                    trans_sentence = ' '.join(trans_tokens)

                    # Detailed Trans
                    detailed_trans_file.writelines(f'Cand.{path_index}: log_prob={lprob:.3f}, score={score:.3f}\n')
                    detailed_trans_file.writelines(trans_sentence + '\n')

                    # Final Trans
                    if self.remove_bpe:
                        trans_sentence = (trans_sentence + ' ').replace(self.bpe_symbol, '').strip()
                    if self.dehyphenate:
                        trans_sentence = dehyphenate(trans_sentence)
                    if path_index == 0:
                        trans_file.writelines(trans_sentence + '\n')
                detailed_trans_file.writelines(f'===================================\n')

    def report(self):
        bleu_scorer = BLEUScorer()
        bleu_scorer.initialize()
        with open(self.trans_path, 'r', encoding='utf-8') as trans_file, open(self.reference_path, 'r', encoding='utf-8') as reference_file:
            for trans_sentence, reference_sentence in zip(trans_file, reference_file):
                bleu_scorer.add(trans_sentence.lower().split(), [reference_sentence.lower().split(), ])

        bleu_score = bleu_scorer.result_string
        self.logger.info('   ' + bleu_score)
        with open(self.detailed_trans_path, 'a', encoding='utf-8') as detailed_trans_file:
            detailed_trans_file.writelines(bleu_score)