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


import torch

from ynmt.testers import register_tester, Tester
from ynmt.testers.ancillaries import BeamSearcher

from ynmt.data.batch import Batch
from ynmt.data.instance import Instance
from ynmt.data.iterator import RawTextIterator
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.sequence import stringize, numericalize, tokenize
from ynmt.utilities.extractor import get_tiled_tensor
from ynmt.utilities.statistics import BLEUScorer


@register_tester('simul_seq2seq')
class SimulSeq2Seq(Tester):
    def __init__(self,
        task, output_names,
        searcher, bpe_symbol, remove_bpe,
        source_path, target_path,
        batch_size, batch_type,
        wait_time,
        device_descriptor, logger
    ):
        super(SimulSeq2Seq, self).__init__(task, output_names, device_descriptor, logger)
        self.searcher = searcher
        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe

        self.source_path = source_path
        self.target_path = target_path
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.wait_time = wait_time

    def initialize(self):
        self.total_sentence_number = 0

    @classmethod
    def setup(cls, args, task, device_descriptor, logger):
        searcher = None
        if args.searcher.name == 'beam':
            searcher = BeamSearcher(
                reserved_path_number = args.searcher.beam_size,
                candidate_path_number = args.searcher.n_best,
                search_space_size = len(task.vocabularies['target']),
                initial_node = task.vocabularies['target'].bos_index,
                terminal_node = task.vocabularies['target'].eos_index,
                min_depth = args.searcher.min_length, max_depth = args.searcher.max_length,
                alpha = args.searcher.penalty.alpha, beta = args.searcher.penalty.beta
            )

        output_names = ['trans', 'trans-detailed']

        simul_seq2seq = cls(
            task, output_names,
            searcher, args.bpe_symbol, args.remove_bpe,
            args.source, args.target,
            args.batch_size, args.batch_type,
            args.wait_time,
            device_descriptor, logger
        )

        return simul_seq2seq

    def test(self, model, batch):
        source = batch.source
        parallel_line_number, _ = source.size()
        self.searcher.initialize(parallel_line_number, self.device_descriptor)

        read_end_position = source.shape[1] - 1
        if self.wait_time == -1:
            read_start_position = read_end_position
        else:
            read_start_position = min(self.wait_time + 1, read_end_position) # +1 for the bos token. When wait_time is 0, first read bos token

        read_position = read_start_position

        while read_start_position <= read_position and read_position <= read_end_position:
            partial_source = torch.index_select(source[:, :read_position + 1], 0, self.searcher.line_original_indices)

            partial_source_mask = model.get_source_mask(partial_source)
            partial_codes = model.encoder(partial_source, partial_source_mask)

            partial_source_mask = get_tiled_tensor(partial_source_mask, 0, self.searcher.reserved_path_number)
            partial_codes = get_tiled_tensor(partial_codes, 0, self.searcher.reserved_path_number)

            if read_position == read_end_position:
                write_until_finished = True
            else:
                write_until_finished = False

            while not self.searcher.finished:
                previous_prediction = self.searcher.found_nodes.reshape(
                    self.searcher.parallel_line_number * self.searcher.reserved_path_number,
                    -1
                )
                previous_prediction_mask = model.get_target_mask(previous_prediction)

                hidden, cross_attention_weight = model.decoder(
                    previous_prediction,
                    partial_codes,
                    previous_prediction_mask,
                    partial_source_mask
                )

                logits = model.generator(hidden)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                prediction_distribution = log_probs[:, -1, :].reshape(
                    self.searcher.parallel_line_number,
                    self.searcher.reserved_path_number,
                    -1
                )
                self.searcher.search(prediction_distribution)

                self.searcher.update()

                partial_source_mask = partial_source_mask.index_select(0, self.searcher.path_offset.reshape(-1))
                partial_codes = partial_codes.index_select(0, self.searcher.path_offset.reshape(-1))

                if write_until_finished:
                    continue
                else:
                    break

            read_position += 1
            if self.searcher.finished:
                break

    def input(self):
        def instance_handler(lines):
            (source_line, ) = lines
            instance = Instance(set({'source', }))
            instance['source'] = numericalize(tokenize(source_line), self.task.vocabularies['source'])
            return instance

        def instance_size_calculator(instances):
            self.max_source_length = 0
            if self.batch_type == 'sentence':
                batch_size = len(instances)

            if self.batch_type == 'token':
                if len(instances) == 1:
                    self.max_source_length = 0

                self.max_source_length = max(self.max_source_length, len(instances[-1].source))

                batch_size = len(instances) * self.max_source_length

            return batch_size
 
        input_iterator = RawTextIterator(
            [self.source_path, ],
            instance_handler,
            self.batch_size,
            instance_size_calculator = instance_size_calculator
        )

        for batch in input_iterator:
            padded_batch = Batch(set({'source', }))
            padded_attributes, _ = pad_attribute(batch.source, self.task.vocabularies['source'].pad_index)
            padded_batch.source = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)
            yield padded_batch

    def output(self, output_basepath):
        results = self.searcher.candidate_paths

        with open(output_basepath + '.' + 'trans', 'a', encoding='utf-8') as translation_file,\
            open(output_basepath + '.' + 'trans-detailed', 'a', encoding='utf-8') as detailed_translation_file:
            for result in results:
                detailed_translation_file.writelines(f'No.{self.total_sentence_number}:\n')
                self.total_sentence_number += 1
                for index, candidate_result in enumerate(result):
                    log_prob = candidate_result['log_prob']
                    score = candidate_result['score']
                    prediction = candidate_result['path']
                    detailed_translation_file.writelines(f'Cand.{index}: log_prob={log_prob:.3f}, score={score:.3f}\n')
                    tokens = stringize(prediction, self.task.vocabularies['target'])
                    sentence = ' '.join(tokens)
                    if self.remove_bpe:
                        sentence = (sentence + ' ').replace(self.bpe_symbol, '').strip()
                    detailed_translation_file.writelines(sentence + '\n')
                    if index == 0:
                        translation_file.writelines(sentence + '\n')
                detailed_translation_file.writelines(f'===================================\n')

    def report(self, output_basepath):
        if self.target_path is None:
            return

        bleu_scorer = BLEUScorer()
        with open(output_basepath + '.' + 'trans', 'r', encoding='utf-8') as translation_file, open(self.target_path, 'r', encoding='utf-8') as reference_file:
            for tra, ref in zip(translation_file, reference_file):
                bleu_scorer.add(tra.split(), [ref.split(), ])

        self.logger.info(bleu_scorer.result_string)

        with open(output_basepath + '.' + 'trans-detailed', 'a', encoding='utf-8') as detailed_translation_file:
            detailed_translation_file.writelines(bleu_scorer.result_string)
