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


from ynmt.generators import Generator

from ynmt.testers import build_tester

from ynmt.data.batch import Batch
from ynmt.data.attribute import pad_attribute

from ynmt.utilities.extractor import get_tiled_tensor


def build_generator_seq2seq(args,
                            model,
                            vocabularies,
                            output_paths,
                            device_descriptor,
                            logger):

    tester = build_tester(args.tester, vocabularies['target'])

    seq2seq = Seq2Seq(
        args.name,
        model,
        tester,
        vocabularies,
        output_paths,
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
                 device_descriptor,
                 logger):
        super(Seq2Seq, self).__init__(name,
                                      model,
                                      vocabularies,
                                      output_paths,
                                      device_descriptor,
                                      logger)
        self.tester = tester
        self.hypothesis_path = self.output_paths['hypothesis']
        with open(self.hypothesis_path, 'w', encoding='utf-8') as hypothesis_file:
            hypothesis_file.truncate()

        self.total_sentence_number = 0

    def customize_batch(self, batch):
        padded_batch = Batch(set(['source', 'target']))
        # source side
        padded_attributes, _ = pad_attribute(batch.source, self.vocabularies['source'].pad_index)
        padded_batch.source = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

        # target side
        padded_attributes, _ = pad_attribute(batch.target, self.vocabularies['target'].pad_index)
        padded_batch.target = torch.tensor(padded_attributes, dtype=torch.long, device=self.device_descriptor)

        return padded_batch

    def generate_batch(self, customized_batch):
        self.statistics.clear()

        source = customized_batch.source
        target = customized_batch.target

        parallel_line_number, max_source_length = source.size()

        source_mask = self.model.get_source_mask(source)
        codes = self.model.encoder(source, source_mask)

        source_mask = get_tiled_tensor(source_mask, 0, self.tester.reserved_path_number)
        codes = get_tiled_tensor(codes, 0, self.tester.reserved_path_number)

        self.tester.initialize(parallel_line_number, self.device_descriptor)

        while not self.tester.finished:
            codes = codes.index_select(0, self.tester.path_offset.reshape(-1))
            #codes_size = list(codes.size())
            #codes_size[0] = self.tester.parallel_line_number * self.tester.reserved_path_number
            #codes = codes.reshape(codes.size(0) // self.tester.reserved_path_number, -1)
            #codes = torch.index_select(codes, 0, self.tester.active_line_indices)
            #codes = codes.reshape(codes_size)

            source_mask = source_mask.index_select(0, self.tester.path_offset.reshape(-1))
            #source_mask_size = list(source_mask.size())
            #source_mask_size[0] = self.tester.parallel_line_number * self.tester.reserved_path_number
            #source_mask = source_mask.reshape(source_mask.size(0) // self.tester.reserved_path_number, -1)
            #source_mask = torch.index_select(source_mask, 0, self.tester.active_line_indices)
            #source_mask = source_mask.reshape(source_mask_size)

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
            #print(prediction_distribution.size())
            self.tester.search(prediction_distribution)
        self.output()
        return

    def output(self):
        results = self.tester.candidate_paths
        with open(self.hypothesis_path, 'a', encoding='utf-8') as hypothesis_file:
            for result in results:
                hypothesis_file.writelines(f'No.{self.total_sentence_number}:\n')
                self.total_sentence_number += 1
                for index, candidate_result in enumerate(result):
                    log_prob = candidate_result['log_prob']
                    score = candidate_result['score']
                    prediction = candidate_result['path']
                    output_line = f'    Cand.{index}: log_prob={log_prob}, score={score} |||'
                    sentence = str()
                    for token_index in prediction:
                        if token_index == self.vocabularies['target'].eos_index:
                            break
                        sentence = sentence + ' ' + self.vocabularies['target'].token(token_index)
                    output_line = output_line + ' ' + sentence + '\n'
                    hypothesis_file.writelines(output_line)
                hypothesis_file.writelines(f'\n')
