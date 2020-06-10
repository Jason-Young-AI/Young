#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-04-02 08:23
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import collections
import multiprocessing


import ynmt.hocon.arguments as harg
import ynmt.utilities.logging as logging
from ynmt.data.vocabulary import Vocabulary
from ynmt.data.instance import Instance
from ynmt.data.dataset import Dataset
from ynmt.utilities.file import file_slice_edges, get_coedges, save_data_objects
from ynmt.utilities.line import numericalize


def get_partial_token_counter(file_path, edge_start, edge_end):
    token_counter = collections.Counter()
    with open(file_path, 'r', encoding='utf-8') as file_object:
        file_object.seek(edge_start)
        while file_object.tell() < edge_end:
            line = file_object.readline()
            token_list = line.strip().split()
            token_counter.update(token_list)
    return token_counter


def get_token_counter(file_path, number_worker):
    total_token_counter = collections.Counter()
    edges = file_slice_edges(file_path, number_worker)

    with multiprocessing.Pool(number_worker) as pool:
        results = list()
        for edge_start, edge_end in edges:
            result = pool.apply_async(get_partial_token_counter,
                                      (
                                          file_path,
                                          edge_start,
                                          edge_end,
                                      ))
            results.append(result)
        for result in results:
            total_token_counter.update(result.get())
    return total_token_counter


def build_vocabulary(args, source_path, target_path, number_worker, logger):
    source_token_counter = get_token_counter(source_path, number_worker)
    target_token_counter = get_token_counter(target_path, number_worker)
    if args.vocabulary.share:
        logger.info(' * Share Vocabulary')
        token_counter = collections.Counter()
        token_counter.update(source_token_counter)
        token_counter.update(target_token_counter)
        vocabulary = Vocabulary(list(token_counter.items()), args.vocabulary.size)
        source_vocabulary = vocabulary
        target_vocabulary = vocabulary
    else:
        source_vocabulary = Vocabulary(list(source_token_counter.items()), args.vocabulary.source_size)
        target_vocabulary = Vocabulary(list(target_token_counter.items()), args.vocabulary.target_size)

    logger.info(f' * source vocabulary {len(source_vocabulary)} token')
    logger.info(f' * target vocabulary {len(target_vocabulary)} token')

    return source_vocabulary, target_vocabulary

def build_instances(source_path, target_path,
                    source_edge, target_edge,
                    source_vocabulary, target_vocabulary,
                    structure):
    instances = list()
    source_edge_start, source_edge_end = source_edge
    target_edge_start, target_edge_end = target_edge
    with open(source_path, 'r', encoding='utf-8') as source_file, open(target_path, 'r', encoding='utf-8') as target_file:
        source_file.seek(source_edge_start)
        target_file.seek(target_edge_start)
        while source_file.tell() < source_edge_end:
            source_line = source_file.readline()
            target_line = target_file.readline()
            instance = Instance(structure)
            instance.source = numericalize(source_line, source_vocabulary)
            instance.target = numericalize(target_line, target_vocabulary)
            instances.append(instance)
    return instances


def get_partial_dataset(source_path, target_path,
                        source_vocabulary, target_vocabulary,
                        source_edges, target_edges,
                        edge_index_start, edge_index_end,
                        number_worker):
    structure = set({'source', 'target'})
    partial_dataset = Dataset(structure)

    def add_instances(instances):
        for instance in instances:
            partial_dataset.add(instance)

    with multiprocessing.Pool(number_worker) as pool:
        results = list()
        for edge_index in range(edge_index_start, edge_index_end):
            result = pool.apply_async(build_instances,
                                      (
                                          source_path, target_path, 
                                          source_edges[edge_index], target_edges[edge_index],
                                          source_vocabulary, target_vocabulary,
                                          structure,
                                      ),
                                      callback=add_instances)
            results.append(result)
        for result in results:
            result.get()

    return partial_dataset

def build_dataset(args, dataset_name,
                  source_path, target_path,
                  source_vocabulary, target_vocabulary,
                  number_worker, number_slice,
                  logger):
    source_edges = file_slice_edges(source_path, number_worker * number_slice)
    target_edges = get_coedges(source_path, source_edges, target_path)
    logger.info(f' * {dataset_name} will be slicing to {number_slice} partial_datasets!')
    datasets = list()
    for index in range(number_slice):
        edge_index_start = index * number_worker
        edge_index_end = edge_index_start + number_worker

        logger.info(f' * Building No.{index} partial_dataset ...')
        dataset = get_partial_dataset(source_path, target_path,
                                      source_vocabulary, target_vocabulary,
                                      source_edges, target_edges,
                                      edge_index_start, edge_index_end,
                                      number_worker)

        logger.info(f' * {len(dataset)} instances  Built.')
        datasets.append(dataset)
    logger.info(f' * Total {sum(len(dataset) for dataset in datasets)} instances.')
    return datasets


def preprocess(args):
    logger = logging.get_logger(logging_path=args.logging_path)

    data_directory = args.data_directory
    if not os.path.isdir(data_directory):
        logger.error('Data directory does not exists!')
        return

    source_language = args.language.source
    target_language = args.language.target
    logger.info(f'source language: {source_language}')
    logger.info(f'target language: {target_language}')

    logger.info('Building vocabulary ...')
    source_vocabulary, target_vocabulary = build_vocabulary(args, args.corpora.train_path.source, args.corpora.train_path.target, args.number_worker, logger)

    logger.info(' > Saving vocabulary ...')
    source_vocab_path = os.path.join(data_directory, f'{source_language}-{target_language}.{source_language}.vocab')
    target_vocab_path = os.path.join(data_directory, f'{source_language}-{target_language}.{target_language}.vocab')
    save_data_objects(source_vocab_path, [source_vocabulary, ])
    logger.info(f' > Saved {source_vocab_path} !')
    save_data_objects(target_vocab_path, [target_vocabulary, ])
    logger.info(f' > Saved {target_vocab_path} !')

    logger.info('Building training dataset ...')
    training_dataset = build_dataset(args, 'train',
                                     args.corpora.train_path.source, args.corpora.train_path.target,
                                     source_vocabulary, target_vocabulary,
                                     args.number_worker, args.number_slice,
                                     logger)

    logger.info(' > Saving training dataset ...')
    training_dataset_name = f'train.{source_language}-{target_language}.dataset'
    training_dataset_path = os.path.join(data_directory, training_dataset_name)
    save_data_objects(training_dataset_path, training_dataset)
    logger.info(f' > Saved {training_dataset_path} !')

    logger.info('Building validation dataset ...')
    validation_dataset = build_dataset(args, 'valid',
                                       args.corpora.valid_path.source, args.corpora.valid_path.target,
                                       source_vocabulary, target_vocabulary,
                                       1, 1,
                                       logger)

    logger.info(' > Saving validation dataset ...')
    validation_dataset_name = f'valid.{source_language}-{target_language}.dataset'
    validation_dataset_path = os.path.join(data_directory, validation_dataset_name)
    save_data_objects(validation_dataset_path, validation_dataset)
    logger.info(f' > Saved {validation_dataset_path} !')

    logger.info(' = Finished !')


def main():
    args = harg.get_arguments()
    preprocess_args = harg.get_partial_arguments(args, 'binaries.preprocess')
    preprocess(preprocess_args)


if __name__ == '__main__':
    main()
