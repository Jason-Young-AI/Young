#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-29 18:33
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
import torch


from ynmt.generators import Generator

from ynmt.testers import build_tester

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.extractor import get_tiled_tensor
from ynmt.utilities.statistics import BLEUScorer


def build_generator_seq2seq(args,
                            model,
                            vocabularies,
                            output_paths,
                            reference_paths,
                            device_descriptor,
                            logger):

    tester = build_tester(args.tester, vocabularies['target'])

    seq2seq = Seq2Seq(
        args.name,
        model,
        tester,
        vocabularies,
        output_paths,
        reference_paths,
        args.bpe_symbol,
        args.remove_bpe,
        device_descriptor,
        logger,
    )
    return seq2seq


class Seq2Seq(Generator):
    def __init__(self,
                 name,
                 model,
                 tester,
                 vocabularies,
                 output_paths,
                 reference_paths,
                 bpe_symbol,
                 remove_bpe,
                 device_descriptor,
                 logger):
        super(Seq2Seq, self).__init__(name,
                                      model,
                                      vocabularies,
                                      output_paths,
                                      reference_paths,
                                      device_descriptor,
                                      logger)
        self.tester = tester
        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe

        self.target_hyp_file =  open(self.output_paths['target_hyp'], 'w', encoding='utf-8')
        self.target_hyp_file.truncate()

        reference_name_pattern = re.compile(f'target_ref(\d+)')
        self.target_ref_files = list()
        for reference_name, reference_path in self.reference_paths.items():
            result = reference_name_pattern.fullmatch(reference_name)
            if result is not None:
                target_ref_file = open(reference_path, 'r', encoding='utf-8')
                self.target_ref_files.append(target_ref_file)

        self.total_sentence_number = 0

        self.bleu_scorer = BLEUScorer()

    def customize_batch(self, batch):
        padded_batch = Batch(set(['source']))
        # source side
        padded_attributes, _ = pad_attribute(batch.source, self.vocabularies['source'].pad_index)
        padded_batch.source = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

        return padded_batch

    def generate_batch(self, customized_batch):
        self.statistics.clear()

        source = customized_batch.source

        parallel_line_number, max_source_length = source.size()

        source_mask = self.model.get_source_mask(source)
        codes = self.model.encoder(source, source_mask)

        source_mask = get_tiled_tensor(source_mask, 0, self.tester.reserved_path_number)
        codes = get_tiled_tensor(codes, 0, self.tester.reserved_path_number)

        self.tester.initialize(parallel_line_number, self.device_descriptor)

        while not self.tester.finished:
            codes = codes.index_select(0, self.tester.path_offset.reshape(-1))

            source_mask = source_mask.index_select(0, self.tester.path_offset.reshape(-1))

            previous_prediction = self.tester.found_nodes.reshape(
                self.tester.parallel_line_number * self.tester.reserved_path_number,
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
                self.tester.parallel_line_number,
                self.tester.reserved_path_number,
                -1
            )
            self.tester.search(prediction_distribution)
        self.output()
        return

    def output(self):
        results = self.tester.candidate_paths
        for result in results:
            ref_sentences = list()
            for target_ref_file in self.target_ref_files:
                ref_sentences.append(target_ref_file.readline().split())
            self.target_hyp_file.writelines(f'No.{self.total_sentence_number}:\n')
            self.total_sentence_number += 1
            for index, candidate_result in enumerate(result):
                log_prob = candidate_result['log_prob']
                score = candidate_result['score']
                prediction = candidate_result['path']
                self.target_hyp_file.writelines(f'Cand.{index}: log_prob={log_prob:.3f}, score={score:.3f}\n')
                token_strings = list()
                for token_index in prediction:
                    if token_index == self.vocabularies['target'].eos_index:
                        break
                    token_strings.append(self.vocabularies['target'].token(token_index))
                sentence = ' '.join(token_strings)
                if self.remove_bpe:
                    sentence = (sentence + ' ').replace(self.bpe_symbol, '').strip()
                if index == 0:
                    self.bleu_scorer.add(sentence.split(), ref_sentences)
                self.target_hyp_file.writelines(sentence + '\n')
            self.target_hyp_file.writelines(f'===================================\n')

    def final_operation(self):
        self.target_hyp_file.writelines(self.bleu_scorer.result_string)
        print(self.bleu_scorer.result_string)
