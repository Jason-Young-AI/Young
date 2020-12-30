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

from ynmt.factories import register_factory, Factory
from ynmt.factories.mixins import SeqMixin

from ynmt.data.vocabulary import Vocabulary
from ynmt.data.iterator import Iterator, RawIterator
from ynmt.data.instance import Instance

from ynmt.utilities.sequence import tokenize, numericalize


def get_instance_size_calculator(batch_type):
    if batch_type == 'sentence':

        def instance_size_calculator(instances):
            batch_size = len(instances)
            return batch_size

    if batch_type == 'token':

        def instance_size_calculator(instances):
            global max_source_length, max_target_length
            if len(instances) == 1:
                max_source_length = 0
                max_target_length = 0
            max_source_length = max(max_source_length, len(instances[-1].source))
            max_target_length = max(max_target_length, len(instances[-1].target) - 1)
            source_batch_size = len(instances) * max_source_length
            target_batch_size = len(instances) * max_target_length

            batch_size = max(source_batch_size, target_batch_size)
            return batch_size

    return instance_size_calculator


@register_factory('bilingual_with_auxinf')
class BilingualWithAuxinf(Factory, SeqMixin):
    def __init__(self, logger, source_language, target_language, auxinf_language):
        super(BilingualWithAuxinf, self).__init__(logger, set({'source', 'target', 'auxinf'}))

        assert auxinf_language in set({source_language, target_language}), f'Invalid auxiliary information language: {auxinf_language}!'

        self.vocabularies = dict()

        self.source_language = source_language
        self.target_language = target_language
        self.auxinf_language = auxinf_language

    @classmethod
    def setup(cls, settings, logger):
        args = settings.args
        return cls(logger, args.language.source, args.language.target, args.language.auxinf)

    def training_batches(self, args):
        source_filter = args.training_batches.filter.source
        target_filter = args.training_batches.filter.target

        def instance_filter(instance):
            if len(instance.source) - 2 < source_filter[0] or source_filter[1] < len(instance.source) - 2:
                return True
            if len(instance.target) - 2 < target_filter[0] or target_filter[1] < len(instance.target) - 2:
                return True
            return False

        def instance_comparator(instance):
            return (len(instance.source), len(instance.target))

        return Iterator(
            args.datasets.training,
            args.training_batches.batch_size,
            dock_size = args.training_batches.dock_size,
            export_volume = args.training_batches.export_volume,
            instance_filter = instance_filter,
            instance_comparator = instance_comparator,
            instance_size_calculator = get_instance_size_calculator(args.training_batches.batch_type),
            shuffle=args.training_batches.shuffle,
            infinite=True,
        )

    def validation_batches(self, args):
        return Iterator(
            args.datasets.validation,
            args.validation_batches.batch_size,
            instance_size_calculator = get_instance_size_calculator(args.validation_batches.batch_type),
        )

    def testing_batches(self, args):
        return RawIterator(
            [args.raw_data.testing.source, args.raw_data.testing.target, args.raw_data.testing.auxinf],
            ['text', 'text', 'text'],
            self.build_instance,
            args.testing_batches.batch_size,
            instance_size_calculator = get_instance_size_calculator(args.testing_batches.batch_type)
        )

    def load_ancillary_datasets(self, args):
        self.logger.info(f'  . Loading vocabularies ...')
        self.vocabularies = load_data(args.datasets.vocabularies)
        vocabulary_sizes = {vocab_name: len(vocabulary) for vocab_name, vocabulary in self.vocabularies.items()}
        self.logger.info(f'  ..Loaded Vocabularies: {vocabulary_sizes}')

    def build_ancillary_datasets(self, args):
        self.logger.info(f'  . Building vocabularies ...')
        self.build_vocabularies(args)

    def build_vocabularies(self, args):
        self.logger.info(f'  ..[\'Source\']')
        self.logger.info(f'    corpus: {args.raw_data.training.source}')
        self.logger.info(f'    language: {self.source_language}')
        source_token_counter = self.count_tokens_of_corpus(args.raw_data.training.source, args.number_worker, args.work_amount)
        self.logger.info(f'    {len(source_token_counter)} token found')

        self.logger.info(f'  ..[\'Target\']')
        self.logger.info(f'    corpus: {args.raw_data.training.target}')
        self.logger.info(f'    language: {self.target_language}')
        target_token_counter = self.count_tokens_of_corpus(args.raw_data.training.target, args.number_worker, args.work_amount)
        self.logger.info(f'    {len(target_token_counter)} token found')

        if args.vocabularies.share:
            merged_token_counter = collections.Counter()
            merged_token_counter.update(source_token_counter)
            merged_token_counter.update(target_token_counter)

            shared_vocabulary_size_limit = min(args.vocabularies.size_limit.source, args.vocabularies.size_limit.target)
            self.logger.info(f'  ..Shared vocabulary will be built within limits of size: {shared_vocabulary_size_limit}')
            special_tokens = args.vocabularies.special_tokens.source + args.vocabularies.special_tokens.target
            shared_vocabulary = Vocabulary(list(merged_token_counter.items()), shared_vocabulary_size_limit, special_tokens=special_tokens)
            self.logger.info(f'    Shared vocabulary size is {len(shared_vocabulary)}')

            source_vocabulary = shared_vocabulary
            target_vocabulary = shared_vocabulary
        else:
            self.logger.info(f'  ..Source vocabulary will be built within limits of size: {args.vocabularies.size_limit.source}')
            source_vocabulary = Vocabulary(list(source_token_counter.items()), args.vocabularies.size_limit.source, special_tokens=args.vocabularies.special_tokens.source)
            self.logger.info(f'    Source vocabulary size is {len(source_vocabulary)}')

            self.logger.info(f'  ..Target vocabulary will be built within limits of size: {args.vocabularies.size_limit.target}')
            target_vocabulary = Vocabulary(list(target_token_counter.items()), args.vocabularies.size_limit.target, special_tokens=args.vocabularies.special_tokens.target)
            self.logger.info(f'    Target vocabulary size is {len(target_vocabulary)}')


        self.vocabularies['source'] = source_vocabulary
        self.vocabularies['target'] = target_vocabulary

        if self.auxinf_language == self.source_language:
            self.vocabularies['auxinf'] = source_vocabulary
        if self.auxinf_language == self.target_language:
            self.vocabularies['auxinf'] = target_vocabulary

        self.logger.info(f'  ..Saving vocabularies to {args.datasets.vocabularies} ...')
        dump_data(args.datasets.vocabularies, self.vocabularies)
        if args.vocabularies.save_readable:
            for name, vocabulary in self.vocabularies.items():
                readable_path = args.datasets.vocabularies + f'.{name}-readable'
                with open(readable_path, 'w', encoding='utf-8') as readable_file:
                    for token, frequency in vocabulary:
                        readable_file.writelines(f'{token} {frequency}\n')
        self.logger.info(f'    Vocabularies has been saved.')

    def align_and_partition_raw_data(self, raw_data_args, partition_size):
        source_corpus_partitions = load_plain(raw_data_args.source, partition_unit='line', partition_size=partition_size)
        target_corpus_partitions = load_plain(raw_data_args.target, partition_unit='line', partition_size=partition_size)
        auxinf_corpus_partitions = load_plain(raw_data_args.auxinf, partition_unit='line', partition_size=partition_size)
        corpus_partitions = zip(source_corpus_partitions, target_corpus_partitions, auxinf_corpus_partitions)
        return corpus_partitions

    def build_instance(self, aligned_raw_data_item):
        source_line, target_line, auxinf_line = aligned_raw_data_item

        source_attribute = numericalize(tokenize(source_line), self.vocabularies['source'], add_bos=True, add_eos=True)
        target_attribute = numericalize(tokenize(target_line), self.vocabularies['target'], add_bos=True, add_eos=True)
        auxinf_attribute = numericalize(tokenize(auxinf_line), self.vocabularies['auxinf'], add_bos=True, add_eos=True)
 
        return Instance(source=source_attribute, target=target_attribute)
