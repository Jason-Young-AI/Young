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

from yoolkit.timer import Timer

from youngs.testers import register_tester, Tester
from youngs.testers.ancillaries import GreedySearcher

from youngs.data.batch import Batch
from youngs.data.instance import Instance
from youngs.data.attribute import pad_attribute

from youngs.utilities.metrics import BLEUScorer
from youngs.utilities.sequence import stringize, numericalize, tokenize, dehyphenate
from youngs.utilities.extractor import get_tiled_tensor, get_foresee_mask, get_padding_mask


@register_tester('wait_k')
class WaitK(Tester):
    def __init__(self,
        factory, model,
        greedy_searcher,
        using_cache, using_uni,
        bpe_symbol, remove_bpe, dehyphenate,
        reference_paths,
        wait_source_time,
        output_directory, output_name,
        device_descriptor, logger
    ):
        super(WaitK, self).__init__(factory, model, output_directory, output_name, device_descriptor, logger)
        self.greedy_searcher = greedy_searcher

        self.using_cache = using_cache
        self.using_uni = using_uni

        self.bpe_symbol = bpe_symbol
        self.remove_bpe = remove_bpe
        self.dehyphenate = dehyphenate
        self.reference_paths = reference_paths

        self.wait_source_time = wait_source_time
        self.test_timer = Timer()
        self.test_timer.reset()
        self.test_timer.launch()
        self.speed = list()

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
            args.using_cache, args.using_uni,
            args.bpe_symbol, args.remove_bpe, args.dehyphenate,
            args.reference_paths,
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

        self.rw_path = output_basepath + '.rw'
        with open(self.rw_path, 'w', encoding='utf-8') as rw_file:
            rw_file.truncate()

    def customize_batch(self, batch):
        padded_source_attributes, _ = pad_attribute(batch.source, self.factory.vocabularies['source'].pad_index)
        source = torch.tensor(padded_source_attributes, dtype=torch.long, device=self.device_descriptor)
        return source

    def test_batch(self, customized_batch):
        original_source = customized_batch
        parallel_line_number, max_source_length = original_source.size()
        source_lengths = ((~get_padding_mask(original_source, self.factory.vocabularies['source'].pad_index)).sum(-1) - 2).tolist()

        if self.using_cache:
            self.model.decoder.clear_caches()

        self.greedy_searcher.initialize(parallel_line_number, self.device_descriptor)

        read_length = self.wait_source_time + 1 # + 1 for bos
        write_time = 0
        self.test_timer.lap()
        while not self.greedy_searcher.finished:
            source = torch.index_select(original_source, 0, self.greedy_searcher.line_original_indices)
            source = source[:, :read_length]
            cross_attention_weight_mask = get_padding_mask(source, self.factory.vocabularies['source'].pad_index).unsqueeze(1)
            if self.using_uni:
                source_mask = get_padding_mask(source, self.factory.vocabularies['source'].pad_index).unsqueeze(1)
                foresee_mask = get_foresee_mask(
                    source.size(-1), source.size(-1),
                    source.device,
                ).unsqueeze(0)
                source_mask = source_mask | foresee_mask
                codes = self.model.encoder(source, source_mask)
            else:
                source_mask = get_padding_mask(source, self.factory.vocabularies['source'].pad_index).unsqueeze(1)
                codes = self.model.encoder(source, source_mask)

            if self.using_cache:
                self.model.decoder.update_caches(self.greedy_searcher.active_line_indices)
                target = self.greedy_searcher.current_nodes.unsqueeze(-1)
            else:
                target = self.greedy_searcher.found_nodes

            target_mask = self.model.get_target_mask(target)

            hidden, cross_attention_weight = self.model.decoder(
                target,
                codes,
                target_mask,
                cross_attention_weight_mask,
                using_step_cache=self.using_cache,
                using_self_cache=self.using_cache,
                using_cross_cache=False,
            )

            logits = self.model.generator(hidden)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            prediction_distribution = log_probs[:, -1, :]
            self.greedy_searcher.search(prediction_distribution)
            self.greedy_searcher.update()

            read_length += 1
            write_time += 1

        self.speed.append((self.test_timer.lap(), write_time))
        return source_lengths, self.greedy_searcher.result

    def output_result(self, result):
        source_lengths, result = result
        parallel_line_number = len(result)
        assert len(source_lengths) == parallel_line_number

        with open(self.trans_path, 'a', encoding='utf-8') as trans_file, \
            open(self.rw_path, 'a', encoding='utf-8') as rw_file:
            for line_index in range(parallel_line_number):
                trans = result[line_index]['path']
                initial_rw = ['0' for i in range(min(self.wait_source_time, source_lengths[line_index]))]

                trans_tokens = stringize(trans, self.factory.vocabularies['target'])

                source_length = max(source_lengths[line_index] - self.wait_source_time, 0)
                target_length = len(trans_tokens)
                read_number = min(source_length, target_length)
                write_number = max(target_length-source_length, 0)
                rw = list(initial_rw)
                for _ in range(read_number):
                    rw.append('1')
                    rw.append('0')
                for _ in range(write_number):
                    rw.append('1')

                trans_sentence = ' '.join(trans_tokens)
                rw_sequence = ' '.join(rw)

                # Final Trans
                if self.remove_bpe:
                    trans_sentence = (trans_sentence + ' ').replace(f'{self.bpe_symbol} ', '').strip()
                if self.dehyphenate:
                    trans_sentence = dehyphenate(trans_sentence)
                trans_file.writelines(trans_sentence + '\n')
                rw_file.writelines(rw_sequence + '\n')

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
        tt = 0
        tn = 0
        for t, n in self.speed:
            tt+=t
            tn+=n

        self.logger.info(f'   Speed: {tt/tn} sec/token')
        self.logger.info(f'   Total_Token = {tn} token')
        self.logger.info(f'   Total_Time = {tt} sec')
