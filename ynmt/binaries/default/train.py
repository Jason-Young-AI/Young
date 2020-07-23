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


import ynmt.hocon.arguments as harg


from ynmt.data.instance import InstanceFilter, InstanceSizeCalculator
from ynmt.data.iterator import Iterator
from ynmt.utilities.file import load_data_objects
from ynmt.utilities.random import fix_random_procedure
from ynmt.utilities.logging import setup_logger, get_logger, logging_level
from ynmt.utilities.visualizing import setup_visualizer, get_visualizer
from ynmt.utilities.extractor import get_model_parameters_number
from ynmt.utilities.checkpoint import load_checkpoint
from ynmt.utilities.distributed import DistributedManager, distributed_main

from ynmt.models import build_model
from ynmt.criterions import build_criterion
from ynmt.testers import build_tester
from ynmt.schedulers import build_scheduler
from ynmt.optimizers import build_optimizer
from ynmt.trainers import build_trainer


def get_device_descriptor(device, process_index):
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{process_index}'

    return torch.device(device_name)


def process_main(args, batch_queue, device_descriptor, workshop_semaphore, rank):
    fix_random_procedure(args.random_seed)

    logger = setup_logger('train', logging_path=args.logging_path, logging_level=logging_level['INFO'])
    visualizer = setup_visualizer(
        args.visualizer.name, args.visualizer.server, args.visualizer.port,
        username=args.visualizer.username,
        password=args.visualizer.password,
        logging_path=args.visualizer.logging_path,
        offline=args.visualizer.offline,
        overwrite=args.visualizer.overwrite,
    )
    is_station = rank == 0
    if is_station:
        logger.disabled = False
        visualizer.disabled = False
    else:
        logger.disabled = True
        visualizer.disabled = True

    def batch_list_receiver():
        batches = list()
        while True:
            batch = batch_queue.get()
            if batch is None:
                return batches
            else:
                workshop_semaphore.release()
                batches.append(batch)

    def batch_generator_receiver():
        while True:
            batch = batch_queue.get()
            if batch is None:
                return
            else:
                workshop_semaphore.release()
                yield batch

    valid_batches = batch_list_receiver()
    train_batches = batch_generator_receiver()

    vocabularies_path = os.path.join(args.data.directory, f'{args.data.name}.vocab')
    vocabularies = list(load_data_objects(vocabularies_path))[0]
    vocabulary_sizes = {side: len(vocabulary) for side, vocabulary in vocabularies.items()}
    logger.info(f' * Loaded Vocabularies: {vocabulary_sizes}')

    logger.info(f' * Checking already exist checkpoint ...')
    checkpoint = load_checkpoint(args.process_control.checkpoint.directory, args.data.name)
    if checkpoint is None:
        logger.info(f'   No checkpoint found in \'{args.process_control.checkpoint.directory}\'.')
    else:
        logger.info(f'   Loaded latest checkpoint from \'{args.process_control.checkpoint.directory}\' at {checkpoint["step"]} steps')

    ## Building Something

    # Build Model
    logger.info(f' * Building Model ...')
    model = build_model(args.model, vocabularies, checkpoint, device_descriptor)
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

    # Build Criterion (Training)
    logger.info(f' * Building Training Criterion ...')
    training_criterion = build_criterion(args.criterion.training, vocabularies['target'])
    logger.info(f'   Training Criterion \'{args.criterion.training.name}\' built.')

    # Build Criterion (Validation)
    logger.info(f' * Building Validation Criterion ...')
    validation_criterion = build_criterion(args.criterion.validation, vocabularies['target'])
    logger.info(f'   Validation Criterion \'{args.criterion.validation.name}\' built.')

    # Build Tester
    logger.info(f' * Building Tester ...')
    tester = build_tester(args.tester, vocabularies['target'])
    logger.info(f'   Tester \'{args.tester.name}\' built.')

    # Build Scheduler
    logger.info(f' * Building Learning Rate Scheduler ...')
    scheduler = build_scheduler(args.scheduler, model, checkpoint)
    logger.info(f'   Scheduler \'{args.scheduler.name}\' built.')

    # Build Optimizer
    logger.info(f' * Building Optimizer ...')
    optimizer = build_optimizer(args.optimizer, model, checkpoint)
    logger.info(f'   Optimizer \'{args.optimizer.name}\' built.')

    # Build Trainer
    logger.info(f' * Building Trainer ...')
    trainer = build_trainer(
        args.trainer,
        model, training_criterion, validation_criterion,
        tester,
        scheduler, optimizer,
        vocabularies,
        device_descriptor,
        checkpoint,
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
        train_batches,
        args.process_control.training.period,
        valid_batches,
        args.process_control.validation.period,
        args.process_control.checkpoint.directory, args.data.name, args.process_control.checkpoint.keep_number
    )

    visualizer.close()
    logger.info(f' * Close Visualizer ...')


def build_batches(args, batch_queues, workshop_semaphore, world_size, ranks):
    fix_random_procedure(args.random_seed)
    logger = setup_logger('train', logging_path=args.logging_path, logging_level=logging_level['INFO'])
    rank2index = dict()
    for index, rank in enumerate(ranks):
        rank2index[rank] = index

    def batch_sender(batch_generator):
        for index, batch in enumerate(batch_generator):
            rank = index % world_size
            if rank not in set(ranks):
                continue
            else:
                workshop_semaphore.acquire()
                batch_queues[rank2index[rank]].put(batch)

        for batch_queue in batch_queues:
            batch_queue.put(None)

    validation_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.valid.dataset')
    validation_batch_generator = Iterator(
        validation_dataset_path,
        args.process_control.validation.batch_size,
        InstanceSizeCalculator(
            args.data.primary_side,
            args.process_control.validation.batch_type
        ),
        InstanceFilter(
            {side: getattr(args.data.filter, side) for side in args.data.filter.sides}
        ) if args.data.filter.validation else None,
    )
    validation_dataset_size = 0
    for partial_validation_dataset in load_data_objects(validation_dataset_path):
        validation_dataset_size += len(partial_validation_dataset)
    logger.info(f' = [Producer] = Validation Dataset contains {validation_dataset_size} Instances.')
    batch_sender(validation_batch_generator)

    training_dataset_path = os.path.join(args.data.directory, f'{args.data.name}.train.dataset')
    training_batch_generator = Iterator(
        training_dataset_path,
        args.process_control.training.batch_size,
        InstanceSizeCalculator(
            args.data.primary_side,
            args.process_control.training.batch_type
        ),
        InstanceFilter(
            {side: getattr(args.data.filter, side) for side in args.data.filter.sides}
        ) if args.data.filter.training else None,
        args.process_control.iteration.traverse_time,
        args.process_control.iteration.mode
    )
    training_dataset_size = 0
    for partial_training_dataset in load_data_objects(training_dataset_path):
        training_dataset_size += len(partial_training_dataset)
    logger.info(f' = [Producer] = Training Dataset contains {training_dataset_size} Instances.')
    batch_sender(training_batch_generator)


def train(args):
    logger = setup_logger('train', logging_path=args.logging_path, logging_level=logging_level['INFO'])

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
        assert torch.cuda.device_count() > number_process, f'Insufficient GPU!'
    logger.info(f'   Master - tcp://{master_ip}:{master_port}')
    logger.info(f'   Ranks({ranks}) in World({world_size})')

    torch.multiprocessing.set_start_method('spawn')
    distributed_manager = DistributedManager()
    distributed_manager.open()
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

    distributed_manager.close()

    logger.info(' $ Finished !')


def main():
    args = harg.get_arguments()
    train_args = harg.get_partial_arguments(args, 'binaries.train')
    train_args.model = harg.get_partial_arguments(args, f'models.{train_args.model}')
    train_args.criterion.training = harg.get_partial_arguments(args, f'criterions.{train_args.criterion.training}')
    train_args.criterion.validation = harg.get_partial_arguments(args, f'criterions.{train_args.criterion.validation}')
    train_args.tester = harg.get_partial_arguments(args, f'testers.{train_args.tester}')
    train_args.scheduler = harg.get_partial_arguments(args, f'schedulers.{train_args.scheduler}')
    train_args.optimizer = harg.get_partial_arguments(args, f'optimizers.{train_args.optimizer}')
    train_args.trainer = harg.get_partial_arguments(args, f'trainers.{train_args.trainer}')
    train(train_args)


if __name__ == '__main__':
    main()
