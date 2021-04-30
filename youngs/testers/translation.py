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


import os
import torch
import sacrebleu

from youngs.testers import register_tester, Tester
from youngs.testers.ancillaries import BeamSearcher

from youngs.data.batch import Batch
from youngs.data.instance import Instance
from youngs.data.attribute import pad_attribute

from youngs.utilities.metrics import BLEUScorer
from youngs.utilities.sequence import stringize, numericalize, tokenize
from youngs.utilities.extractor import get_tiled_tensor, get_foresee_mask, get_padding_mask

from compora.tokenize import split_aggressive_hyphen
from compora.detokenize import merge_aggressive_hyphen, detokenize

from yoolkit.xmlscape import encode
from yoolkit.cio import mk_temp, rm_temp


@register_tester('translation')
class Translation(Tester):
    def __init__(self,
        factory, model,
        beam_searcher,
        using_cache,
        bpe_symbol, remove_bpe, dehyphenate,
        reference_paths,
        sacrebleu_command,
        output_directory, output_name,
        device_descriptor, logger
    ):
        super(Translation, self).__init__(factory, model, output_directory, output_name, device_descriptor, logger)
        self.beam_searcher = beam_searcher

        self.using_cache = using_cache

        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe
        self.dehyphenate = dehyphenate
        self.reference_paths = reference_paths
        self.sacrebleu_command = sacrebleu_command

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
            args.using_cache,
            args.bpe_symbol, args.remove_bpe, args.dehyphenate,
            args.reference_paths,
            args.sacrebleu_command,
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

        self.detokenized_trans_path = output_basepath + '.detokenized_trans'
        with open(self.detokenized_trans_path, 'w', encoding='utf-8') as detokenized_trans_file:
            detokenized_trans_file.truncate()

        self.detailed_trans_path = output_basepath + '.detailed_trans'
        with open(self.detailed_trans_path, 'w', encoding='utf-8') as detailed_trans_file:
            detailed_trans_file.truncate()

    def customize_batch(self, batch):
        padded_source_attributes, _ = pad_attribute(batch.source, self.factory.vocabularies['source'].pad_index)
        source = torch.tensor(padded_source_attributes, dtype=torch.long, device=self.device_descriptor)
        return source

    def test_batch(self, customized_batch):
        original_source = customized_batch
        parallel_line_number, max_source_length = original_source.size()
        source_lengths = ((~get_padding_mask(original_source, self.factory.vocabularies['source'].pad_index)).sum(-1) - 2).tolist()

        source_mask = self.model.get_source_mask(original_source)
        codes = self.model.encoder(original_source, source_mask)

        source_mask = get_tiled_tensor(source_mask, 0, self.beam_searcher.reserved_path_number)
        codes = get_tiled_tensor(codes, 0, self.beam_searcher.reserved_path_number)

        if self.using_cache:
            self.model.decoder.clear_caches()

        self.beam_searcher.initialize(parallel_line_number, self.device_descriptor)

        while not self.beam_searcher.finished:
            source_mask = source_mask.index_select(0, self.beam_searcher.path_offset.reshape(-1))
            codes = codes.index_select(0, self.beam_searcher.path_offset.reshape(-1))
            if self.using_cache:
                self.model.decoder.update_caches(self.beam_searcher.path_offset.reshape(-1))
                target = self.beam_searcher.current_nodes.reshape(
                    self.beam_searcher.parallel_line_number * self.beam_searcher.reserved_path_number,
                    -1
                )
            else:
                target = self.beam_searcher.found_nodes.reshape(
                    self.beam_searcher.parallel_line_number * self.beam_searcher.reserved_path_number,
                    -1
                )

            target_mask = self.model.get_target_mask(target)

            hidden, cross_attention_weight = self.model.decoder(
                target,
                codes,
                target_mask,
                source_mask,
                using_step_cache=self.using_cache,
                using_self_cache=self.using_cache,
                using_cross_cache=self.using_cache,
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

        return source_lengths, self.beam_searcher.candidate_paths

    def output_result(self, result):
        source_lengths, candidate_paths = result
        parallel_line_number = len(candidate_paths)
        assert len(source_lengths) == parallel_line_number

        with open(self.trans_path, 'a', encoding='utf-8') as trans_file, \
            open(self.detokenized_trans_path, 'a', encoding='utf-8') as detokenized_trans_file, \
            open(self.detailed_trans_path, 'a', encoding='utf-8') as detailed_trans_file:
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
                        trans_sentence = (trans_sentence + ' ').replace(f'{self.bpe_symbol} ', '').strip()

                    if self.dehyphenate:
                        trans_sentence = split_aggressive_hyphen(trans_sentence)

                    detokenized_trans_sentence = merge_aggressive_hyphen(trans_sentence)
                    detokenized_trans_sentence = encode(detokenized_trans_sentence)
                    detokenized_trans_sentence = detokenize(detokenized_trans_sentence, self.factory.target_language)

                    if path_index == 0:
                        trans_file.writelines(trans_sentence + '\n')
                        detokenized_trans_file.writelines(detokenized_trans_sentence + '\n')
                detailed_trans_file.writelines(f'===================================\n')

    def report(self):
        bleu_scorer = BLEUScorer()
        bleu_scorer.initialize()
        trans_file = open(self.trans_path, 'r', encoding='utf-8')
        reference_files = list()
        for reference_path in self.reference_paths:
            reference_file = open(reference_path, 'r', encoding='utf-8')
            reference_files.append(reference_file)

        for trans_sentence, reference_sentences in zip(trans_file, zip(*reference_files)):
            ref_list = [reference_sentence.lower().split() for reference_sentence in reference_sentences]
            bleu_scorer.add(trans_sentence.lower().split(), ref_list)

        trans_file.close()
        for reference_file in reference_files:
            reference_file.close()

        bleu_score = bleu_scorer.result_string
        self.logger.info('   ' + bleu_score)

        sacrebleu_temp_path = mk_temp('youngs-sacrebleu_', temp_type='file')
        os.system(f'cat {self.detokenized_trans_path} | sacrebleu {self.sacrebleu_command} > {sacrebleu_temp_path}')
        os.system(f'cat {sacrebleu_temp_path}')
        sacrebleu_signature = ""
        with open(sacrebleu_temp_path, 'r', encoding='utf-8') as sacrebleu_temp_file:
            sacrebleu_signature = sacrebleu_temp_file.readlines()[-1].strip()

        with open(self.detailed_trans_path, 'a', encoding='utf-8') as detailed_trans_file:
            detailed_trans_file.writelines(bleu_score)
            detailed_trans_file.writelines(sacrebleu_signature)

        rm_temp(sacrebleu_temp_path)
