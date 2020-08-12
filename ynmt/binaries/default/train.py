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
import torch
import importlib
import pickle


import ynmt.hocon.arguments as harg


from ynmt.data.iterator import Iterator
from ynmt.data.instance import InstanceFilter, InstanceSizeCalculator, InstanceComparator

from ynmt.utilities.file import load_data_objects
from ynmt.utilities.random import fix_random_procedure
from ynmt.utilities.logging import setup_logger, logging_level
from ynmt.utilities.visualizing import setup_visualizer
from ynmt.utilities.extractor import get_model_parameters_number
from ynmt.utilities.distributed import DistributedManager, distributed_main, distributed_data_sender, distributed_data_receiver, get_device_descriptor

from ynmt.models import build_model
from ynmt.trainers import build_trainer


def process_main(args, batch_queue, device_descriptor, workshop_semaphore, rank):
    fix_random_procedure(args.random_seed)
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])
    visualizer = setup_visualizer(
        args.visualizer.name, args.visualizer.server, args.visualizer.port,
        username=args.visualizer.username,
        password=args.visualizer.password,
        logging_path=args.visualizer.path,
        offline=args.visualizer.offline,
        overwrite=args.visualizer.overwrite,
    )
    is_station = rank == 0
    if is_station:
        logger.disabled = False | args.logger.off
        visualizer.disabled = False | args.visualizer.off
    else:
        logger.disabled = True
        visualizer.disabled = True

    valid_batches = distributed_data_receiver('list', batch_queue, workshop_semaphore)
    train_batches = distributed_data_receiver('generator', batch_queue, workshop_semaphore)

    ## Building Something

    # Build Vocabularies
    vocabularies_path = os.path.join(args.data.directory, f'{args.data.name}.vocab')
    vocabularies = list(load_data_objects(vocabularies_path))[0]
    vocabulary_sizes = {side_name: len(vocabulary) for side_name, vocabulary in vocabularies.items()}
    logger.info(f' * Loaded Vocabularies: {vocabulary_sizes}')

    # Build Model
    logger.info(f' * Building Model ...')
    model = build_model(args.model, vocabularies)
    parameters_number = get_model_parameters_number(model)
    parameters_number_str = str()
    for name, number in parameters_number.items():
        parameters_number_str += f'{name}: {number} Elements ;\n'
    parameters_number_str += f'Total: {sum(parameters_number.values())} Elements .\n'
    logger.info(
        f'\n ~ Model Architecture:'
        f'\n{model}'
        f'\n ~ Number of Parameters:'
        f'\n{parameters_number_str}'
    )

    # Build Trainer
    logger.info(f' * Building Trainer ...')
    trainer = build_trainer(
        args.trainer,
        model,
        vocabularies,
        device_descriptor,
        logger, visualizer,
    )
    logger.info(f'   Trainer \'{args.trainer.name}\' built.')

    # Open Visualizer
    logger.info(f' * Open Visualizer ...')
    visualizer.open()
    if visualizer.offline:
        logger.info(f'   Visualizer in Offline mode')
    else:
        logger.info(f'   Visualizer connection established between {visualizer.server}:{visualizer.port}')
    logger.info(f'   Visualizer logging to \'{visualizer.logging_path}\' .')

    # Launch Trainer
    logger.info(f' * Launch Trainer ...')
    logger.info(f'   Saving checkpoint every {args.process_control.training.period} steps ...')
    logger.info(f'   Validate every {args.process_control.validation.period} steps ...')

    trainer.launch(
        train_batches, args.process_control.training.period,
        valid_batches, args.process_control.validation.period,
    )

    visualizer.close()
    logger.info(f' * Close Visualizer ...')


def build_batches(args, batch_queues, workshop_semaphore, world_size, ranks):
    fix_random_procedure(args.random_seed)
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])

    sides = {side_name: side_tag for side_name, side_tag in args.data.sides}
    side_names = list(sides.keys())
    validation_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.valid.dataset')
    validation_batch_generator = Iterator(
        validation_dataset_path,
        args.process_control.validation.batch_size,
        InstanceSizeCalculator(
            set(side_names),
            args.process_control.validation.batch_type
        ),
        instance_filter=InstanceFilter(
            {side_name: args.data.filter[side_name] for side_name, side_tag in sides.items()}
        ) if args.data.filter.validation else None,
    )
    distributed_data_sender(validation_batch_generator, batch_queues, workshop_semaphore, world_size, ranks)

    training_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.train.dataset')
    training_batch_generator = Iterator(
        training_dataset_path,
        args.process_control.training.batch_size,
        InstanceSizeCalculator(
            set(side_names),
            args.process_control.training.batch_type
        ),
        instance_filter=InstanceFilter(
            {side_name: args.data.filter[side_name] for side_name, side_tag in sides.items()}
        ) if args.data.filter.training else None,
        instance_comparator=InstanceComparator(args.process_control.iteration.sort_order),
        traverse_time=args.process_control.iteration.traverse_time,
        accumulate_number=args.process_control.iteration.accumulate_number,
        mode=args.process_control.iteration.mode
    )
    distributed_data_sender(training_batch_generator, batch_queues, workshop_semaphore, world_size, ranks)


def train(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'])

    device = args.process_control.distribution.device
    master_ip = args.process_control.distribution.master_ip
    master_port = args.process_control.distribution.master_port
    world_size = args.process_control.distribution.world_size
    ranks = args.process_control.distribution.ranks
    workshop_capacity = args.process_control.distribution.workshop_capacity
    number_process = len(ranks)

    if device == 'CPU':
        logger.info(' * Distribution on CPU ...')
    if device == 'GPU':
        logger.info(' * Distribution on GPU ...')
        assert torch.cuda.device_count() >= number_process, f'Insufficient GPU!'
    logger.info(f'   Master - tcp://{master_ip}:{master_port}')
    logger.info(f'   Ranks({ranks}) in World({world_size})')

    torch.multiprocessing.set_start_method('spawn')
    distributed_manager = DistributedManager()
    workshop_semaphore = torch.multiprocessing.Semaphore(world_size * workshop_capacity)

    consumers = list()
    batch_queues = list()
    for process_index in range(number_process):
        device_descriptor = get_device_descriptor(device, process_index)
        batch_queue = torch.multiprocessing.Queue(workshop_capacity)

        main_args = [args, batch_queue, device_descriptor, workshop_semaphore, ranks[process_index]]
        init_args = [device, master_ip, master_port, world_size, ranks[process_index]]
        consumer = torch.multiprocessing.Process(
            target=distributed_main,
            args=(
                process_main,
                main_args,
                init_args,
                distributed_manager.exception_queue,
            ),
            daemon=True
        )
        distributed_manager.manage(consumer)
        consumer.start()
        logger.info(f' * No.{process_index} Training Process start ...')
        logger.info(f'   PID: {consumer.pid};')
        logger.info(f'   Rank: {ranks[process_index]}/{world_size};')
        logger.info(f'   Device: {device_descriptor}.')

        consumers.append(consumer)
        batch_queues.append(batch_queue)

    producer = torch.multiprocessing.Process(
        target=build_batches,
        args=(
            args,
            batch_queues,
            workshop_semaphore,
            world_size,
            ranks,
        ),
        daemon=True
    )
    distributed_manager.manage(producer)
    producer.start()
    logger.info(f' = [Producer] = Batch Producer Process start (PID: {producer.pid}).')

    for consumer in consumers:
        consumer.join()
    producer.join()

    logger.info(' $ Finished !')


def main():
    args = harg.get_arguments()
    train_args = harg.get_partial_arguments(args, 'binaries.train')
    train_args.model = harg.get_partial_arguments(args, f'models.{train_args.model}')
    train_args.trainer = harg.get_partial_arguments(args, f'trainers.{train_args.trainer}')
    train(train_args)


if __name__ == '__main__':
    main()
