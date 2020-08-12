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


from ynmt.data.dataset import Dataset
from ynmt.data.instance import Instance
from ynmt.data.vocabulary import Vocabulary

from ynmt.utilities.file import file_slice_edges, get_coedges, save_data_objects
from ynmt.utilities.line import tokenize, numericalize
from ynmt.utilities.random import fix_random_procedure
from ynmt.utilities.logging import setup_logger, get_logger, logging_level


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


def build_vocabularies(sides, groups, sizes, paths, number_worker, logger_name):
    logger = get_logger(logger_name)
    token_counters = dict()
    for index, (side_name, side_tag) in enumerate(sides.items()):
        path = paths[side_name]
        token_counter = get_token_counter(path, number_worker)
        token_counters[side_name] = token_counter
        logger.info(f' * No.{index} - {side_name}')
        logger.info(f'   corpus: {path}')
        logger.info(f'   tag: {side_tag}')
        logger.info(f'   {len(token_counter)} token found')

    vocabularies = dict()
    for group, size in zip(groups, sizes):
        logger.info(f' * Sides {group} will be merged with size limit {size}')
        token_counter = collections.Counter()
        for side_name in group:
            token_counter.update(token_counters[side_name])
        vocabulary = Vocabulary(list(token_counter.items()), size)
        for side_name in group:
            vocabularies[side_name] = vocabulary
        logger.info(f'   Merged vocabulary size is {len(vocabulary)}')

    return vocabularies


def build_instances(primary_side_name, side_names, paths, vocabularies, corpora_edge):
    structure = set(side_names)
    files = dict()
    for side_name in side_names:
        corpus_edge_start, corpus_edge_end = corpora_edge[side_name]
        files[side_name]= open(paths[side_name], 'r', encoding='utf-8')
        files[side_name].seek(corpus_edge_start)

    instances = list()
    primary_corpus_edge_start, primary_corpus_edge_end = corpora_edge[primary_side_name]
    while files[primary_side_name].tell() < primary_corpus_edge_end:
        instance = Instance(structure)
        for side_name in side_names:
            instance[side_name] = numericalize(tokenize(files[side_name].readline()), vocabularies[side_name])
        instances.append(instance)

    for side_name in side_names:
        files[side_name].close()

    return instances


def get_partial_dataset(primary_side_name, side_names, paths, vocabularies, corpora_edges,
                        edge_index_start, edge_index_end, number_worker):
    structure = set(side_names)
    partial_dataset = Dataset(structure)

    def add_instances(instances):
        for instance in instances:
            partial_dataset.add(instance)

    with multiprocessing.Pool(number_worker) as pool:
        results = list()
        for edge_index in range(edge_index_start, edge_index_end):
            corpora_edge = dict({side_name: corpora_edges[side_name][edge_index] for side_name in side_names})
            result = pool.apply_async(build_instances, (primary_side_name, side_names, paths, vocabularies, corpora_edge))
            results.append(result)
        for result in results:
            add_instances(result.get())

    return partial_dataset


def build_dataset(dataset_type, sides, paths, vocabularies, logger_name,
                  number_worker=1, number_slice=1):
    logger = get_logger(logger_name)

    side_names = list(sides.keys())
    primary_side_name = side_names[0]
    primary_path = paths[primary_side_name]
    primary_edges = file_slice_edges(primary_path, number_worker * number_slice)

    corpora_edges = dict()
    corpora_edges[primary_side_name] = primary_edges
    for side_name in sides.keys():
        if side_name == primary_side_name:
            continue
        path = paths[side_name]
        edges = get_coedges(primary_path, primary_edges, path)
        corpora_edges[side_name] = edges

    logger.info(f' * {dataset_type} will be slicing to {number_slice} partial_datasets!')
    datasets = list()
    for index in range(number_slice):
        edge_index_start = index * number_worker
        edge_index_end = edge_index_start + number_worker

        logger.info(f' ** Building No.{index} partial_dataset ...')
        dataset = get_partial_dataset(primary_side_name, side_names, paths, vocabularies, corpora_edges,
                                      edge_index_start, edge_index_end, number_worker)

        logger.info(f'    {len(dataset)} instances Built.')
        datasets.append(dataset)
    logger.info(f' * Total {sum(len(dataset) for dataset in datasets)} instances.')

    return datasets


def preprocess(args):
    logger = setup_logger(name=args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])

    fix_random_procedure(args.random_seed)

    if not os.path.isdir(args.data.directory):
        logger.error('Data directory does not exists!')
        return

    sides = {side_name: side_tag for side_name, side_tag in args.data.sides}
    logger.info('Building vocabulary ...')
    vocabularies = build_vocabularies(
        sides,
        args.vocabulary.groups,
        args.vocabulary.sizes,
        args.corpus.train_paths,
        args.data.number_worker,
        args.logger.name,
    )

    logger.info(' > Saving vocabularies ...')
    vocabs_path = os.path.join(args.data.directory, f'{args.data.name}.vocab')
    save_data_objects(vocabs_path, [vocabularies, ])
    logger.info(f' > Saved {vocabs_path} !')

    logger.info('Building training dataset ...')
    training_dataset = build_dataset(
        'train',
        sides,
        args.corpus.train_paths,
        vocabularies,
        args.logger.name,
        args.data.number_worker,
        args.data.number_slice,
    )

    logger.info(' > Saving training dataset ...')
    training_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.train.dataset')
    save_data_objects(training_dataset_path, training_dataset)
    logger.info(f' > Saved {training_dataset_path} !')

    logger.info('Building validation dataset ...')
    validation_dataset = build_dataset(
        'valid',
        sides,
        args.corpus.valid_paths,
        vocabularies,
        args.logger.name,
    )

    logger.info(' > Saving validation dataset ...')
    validation_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.valid.dataset')
    save_data_objects(validation_dataset_path, validation_dataset)
    logger.info(f' > Saved {validation_dataset_path} !')

    logger.info(' = Finished !')


def main():
    args = harg.get_arguments()
    preprocess_args = harg.get_partial_arguments(args, 'binaries.preprocess')
    preprocess(preprocess_args)


if __name__ == '__main__':
    main()
