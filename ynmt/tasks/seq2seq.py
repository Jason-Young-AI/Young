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


import collections

from yoolkit.cio import load_plain, load_data, dump_data

from ynmt.tasks import register_task, Task
from ynmt.tasks.mixins import SeqMixin

from ynmt.data.vocabulary import Vocabulary
from ynmt.data.iterator import Iterator
from ynmt.data.instance import Instance

from ynmt.utilities.sequence import tokenize, numericalize


@register_task('seq2seq')
class Seq2Seq(Task, SeqMixin):
    def __init__(self, logger, source_language, target_language):
        super(Seq2Seq, self).__init__(logger, set({'source', 'target'}))

        self.vocabularies = dict()

        self.source_language = source_language
        self.target_language = target_language

    @classmethod
    def setup(cls, settings, logger):
        args = settings.args
        return cls(logger, args.language.source, args.language.target)

    def instance_filter(self, args):
        source_filter = args.filter.source
        target_filter = args.filter.target

        def handler(instance):
            if len(instance.source) < source_filter[0] or source_filter[1] < len(instance.source):
                return True

            if len(instance.target) < target_filter[0] or target_filter[1] < len(instance.target):
                return True

            return False

        return handler

    def instance_comparator(self, args):
        def handler(instance):
            return (len(instance.source), len(instance.target))
        return handler

    def instance_size_calculator(self, args):
        batch_type = args.batch_type
        self.max_source_length = 0
        self.max_target_length = 0

        def handler(instances):
            if batch_type == 'sentence':
                batch_size = len(instances)

            if batch_type == 'token':
                if len(instances) == 1:
                    self.max_source_length = 0
                    self.max_target_length = 0

                self.max_source_length = max(self.max_source_length, len(instances[-1].source))
                self.max_target_length = max(self.max_target_length, len(instances[-1].target) - 1)

                source_batch_size = len(instances) * self.max_source_length
                target_batch_size = len(instances) * self.max_target_length
                batch_size = max(source_batch_size, target_batch_size)

            return batch_size
        
        return handler

    def training_batches(self, args):
        return Iterator(
            args.datasets.training,
            args.training_batches.batch_size,
            dock_size = args.training_batches.dock_size,
            export_volume = args.training_batches.export_volume,
            instance_filter = self.instance_filter(args.training_batches),
            instance_comparator = self.instance_comparator(args.training_batches),
            instance_size_calculator = self.instance_size_calculator(args.training_batches),
            shuffle=args.training_batches.shuffle,
            infinite=True,
        )

    def validation_batches(self, args):
        return Iterator(
            args.datasets.validation,
            args.validation_batches.batch_size,
            instance_size_calculator = self.instance_size_calculator(args.validation_batches),
        )

    def load_ancillary_datasets(self, args):
        self.logger.info(' * Loading vocabularies ...')
        self.vocabularies = load_data(args.datasets.vocabularies)
        vocabulary_sizes = {vocab_name: len(vocabulary) for vocab_name, vocabulary in self.vocabularies.items()}
        self.logger.info(f'   Loaded Vocabularies: {vocabulary_sizes}')

    def build_ancillary_datasets(self, args):
        self.logger.info(f'Building vocabularies ...')
        self.build_vocabularies(args)

    def build_vocabularies(self, args):
        self.logger.info(f' * [\'Source\'] side')
        self.logger.info(f'   corpus: {args.raw_data.training.source}')
        self.logger.info(f'   language: {self.source_language}')
        source_token_counter = self.count_tokens_of_corpus(args.raw_data.training.source, args.number_worker, args.work_amount)
        self.logger.info(f'   {len(source_token_counter)} token found')

        self.logger.info(f' * [\'Target\'] side')
        self.logger.info(f'   corpus: {args.raw_data.training.target}')
        self.logger.info(f'   language: {self.target_language}')
        target_token_counter = self.count_tokens_of_corpus(args.raw_data.training.target, args.number_worker, args.work_amount)
        self.logger.info(f'   {len(target_token_counter)} token found')

        if args.vocabularies.share:
            merged_token_counter = collections.Counter()
            merged_token_counter.update(source_token_counter)
            merged_token_counter.update(target_token_counter)

            shared_vocabulary_size_limit = min(args.vocabularies.size_limit.source, args.vocabularies.size_limit.target)
            self.logger.info(f' * Shared vocabulary will be built within limits of size: {shared_vocabulary_size_limit}')
            shared_vocabulary = Vocabulary(list(merged_token_counter.items()), shared_vocabulary_size_limit)
            self.logger.info(f'   Shared vocabulary size is {len(shared_vocabulary)}')

            source_vocabulary = shared_vocabulary
            target_vocabulary = shared_vocabulary
        else:
            self.logger.info(f' * Source vocabulary will be built within limits of size: {args.vocabularies.size_limit.source}')
            source_vocabulary = Vocabulary(list(source_token_counter.items()), args.vocabularies.size_limit.source)
            self.logger.info(f'   Source vocabulary size is {len(source_vocabulary)}')

            self.logger.info(f' * Target vocabulary will be built within limits of size: {args.vocabularies.size_limit.target}')
            target_vocabulary = Vocabulary(list(target_token_counter.items()), args.vocabularies.size_limit.target)
            self.logger.info(f'   Target vocabulary size is {len(target_vocabulary)}')


        self.vocabularies['source'] = source_vocabulary
        self.vocabularies['target'] = target_vocabulary

        self.logger.info(f' > Saving vocabularies to {args.datasets.vocabularies} ...')
        dump_data(args.datasets.vocabularies, self.vocabularies)
        self.logger.info(f'   Vocabularies has been saved.')

    def align_and_partition_raw_data(self, raw_data_args, partition_size):
        source_corpus_partitions = load_plain(raw_data_args.source, partition_unit='line', partition_size=partition_size)
        target_corpus_partitions = load_plain(raw_data_args.target, partition_unit='line', partition_size=partition_size)
        corpus_partitions = zip(source_corpus_partitions, target_corpus_partitions)
        return corpus_partitions

    def build_instance(self, aligned_raw_data_item):
        source_line, target_line = aligned_raw_data_item

        instance = Instance(self.structure)
        instance['source'] = numericalize(tokenize(source_line), self.vocabularies['source'])
        instance['target'] = numericalize(tokenize(target_line), self.vocabularies['target'])
 
        return instance
