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
import ynmt.utilities.logging as logging


from ynmt.data.instance import InstanceFilter, InstanceSizeCalculator
from ynmt.data.batch import pad_batch
from ynmt.data.iterator import Iterator
from ynmt.utilities.random import fix_random_procedure
from ynmt.utilities.distributed import DistributedManager, distributed_main


def get_device_descriptor(device, process_index):
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{process_index}'

    return torch.device(device_name)


def process_main(args, batch_queue, device_descriptor, workshop_semaphore, rank, logger):
    is_station = rank == 0
    if is_station:
        logger.disabled = False
    else:
        logger.disabled = True

    fix_random_procedure(args.random_seed)

    def batch_receiver(mode):
        assert mode in {'list', 'generator'}, 'Wrong choice of mode.'
        batches = list()
        while True:
            batch = batch_queue.get()
            if batch is None:
                return batches
            else:
                workshop_semaphore.release()
                if mode == 'list':
                    batches.append(batch)
                elif mode == 'generator':
                    yield batch

    valid_batches = batch_receiver('list')
    train_batches = batch_receiver('generator')

    vocabularies_path = os.path.join(args.data.directory, f'{args.data.name}.vocab')
    vocabularies = list(load_data_objects(vocabularies_path))[0]
    vocabulary_sizes = {side: len(vocabulary) for side, vocabulary in vocabularies.items()}
    logger.info(f' * Loaded Vocabularies: {vocabulary_sizes}')

    logger.info(' * Checking already exist checkpoint ...')
    checkpoint = load_checkpoint(args.checkpoint.directory, args.data.name)
    if checkpoint is None:
        logger.info('   No checkpoint found in {args.checkpoint.directory}.')
    else:
        logger.info('   Loaded latest checkpoint from {args.checkpoint.directory} at {checkpoint["step"]} steps')

    logger.info(' * Building Model ...')
    build_model = importlib.import_module(f'ynmt.models.build_model_{args.model.name}')
    model = build_model(args.model, vocabularies, checkpoint)
    model.to(device_descriptor)
    logger.info('   Model Architecture:\n{model}')

    logger.info(' * Building Training Criterion ...')
    build_training_criterion = importlib.import_module(
        f'ynmt.criterions.build_criterion_{args.criterion.training.name}'
    )
    training_criterion = build_training_criterion(args.criterion.training, vocabularies['target'])
    logger.info('   Training Criterion {args.criterion.training.name} built.')

    logger.info(' * Building Validation Criterion ...')
    build_validation_criterion = importlib.import_module(
        f'ynmt.criterions.build_criterion_{args.criterion.validation.name}'
    )
    validation_criterion = build_criterion(args.criterion.validation, checkpoint)
    logger.info('   Validation Criterion {args.criterion.validation.name} built.')

    logger.info(' * Building Tester ...')
    build_tester = importlib.import_module(f'ynmt.tester.build_tester_{args.tester.name}')
    tester = build_tester(args.tester, checkpoint)
    logger.info('   Tester {args.tester.name} built.')

    logger.info(' * Building Learning Rate Scheduler ...')
    build_scheduler = importlib.import_module(f'ynmt.schedulers.build_scheduler_{args.scheduler.name}')
    scheduler = build_scheduler(args.scheduler, checkpoint)
    logger.info('   Scheduler {args.scheduler.name} built.')

    logger.info(' * Building Optimizer ...')
    build_optimizer = importlib.import_module(f'ynmt.optimizers.build_optimizer_{args.optimizer.name}')
    optimizer = build_optimizer(args.optimizer, scheduler, model, checkpoint)
    logger.info('   Optimizer {args.optimizer.name} built.')

    logger.info(' * Building Trainer ...')
    build_trainer = importlib.import_module(f'ynmt.trainers.build_trainer_{args.trainer.name}')
    trainer = build_trainer(args.trainer, model, optimizer, is_station, device_descriptor)
    logger.info('   Trainer {args.trainer.name} built.')

    logger.info(' * Launch Trainer ...')
    logger.info('   Saving checkpoint every {args.process_control.training.period} steps ...')
    logger.info('   Validate every {args.process_control.validation.period} steps ...')
    trainer.launch(
        train_batches,
        args.process_control.training.period,
        valid_batches,
        args.process_control.validation.period,
        args.checkpoint.directory, args.data.name, args.checkpoint.keep_number
    )


def build_batches(args, batch_queues, device_descriptors, workshop_semaphore, world_size):
    assert len(batch_queues) == len(device_descriptors), 'An Error occurred in Data Distribution.'

    rank2index = dict()
    for index, rank in enumerate(ranks):
        rank2index[rank] = index
    distribute_pack = list(zip(batch_queues, device_descriptors))

    def batch_sender(batch_generator):
        for index, batch in enumerate(batch_generator):
            rank = index % world_size
            if rank not in set(ranks):
                continue
            else:
                batch_queue, device_descriptor = distribute_pack[rank2index[rank]]
                pad_batch(batch)
                for attribute_name in batch.structure:
                    (padded_instances, instance_lengths) = batch[attribute_name]
                    padded_instances = torch.tensor(padded_instances, dtype=torch.long, device=device_descriptor)
                    instance_lengths = torch.tensor(instance_lengths, dtype=torch.long, device=device_descriptor)
                    batch[attribute_name] = (padded_instances, instance_lengths)
                workshop_semaphore.acquire()
                batch_queue.put(batch)

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
    logger.info(f'   Validation Dataset contains {validation_dataset_size} Instances.')

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
        args.process_control.traverse_time,
        args.process_control.mode
    )
    training_dataset_size = 0
    for partial_training_dataset in load_data_objects(training_dataset_path):
        training_dataset_size += len(partial_training_dataset)
    logger.info(f'   Training Dataset contains {training_dataset_size} Instances.')

    batch_sender(valid_batch_generator)
    batch_sender(train_batch_generator)


def train(args):
    logger = logging.get_logger(logging_path=args.logging_path)

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
        assert torch.cuda.device_count() < number_process, 'Insufficient GPU!'
    logger.info(f'   Master - tcp://{master_ip}:{master_port}')
    logger.info(f'   Ranks({ranks}) in World({world_size})')

    torch.multiprocessing.set_start_method('spawn')
    distributed_manager = DistributedManager()
    workshop_semaphore = torch.multiprocessing.Semaphore(world_size * workshop_capacity)

    consumers = list()
    batch_queues = list()
    device_descriptors = list()
    for process_index in range(number_process):
        device_descriptor = get_device_descriptor(device, process_index)
        batch_queue = torch.multiprocessing.Queue(workshop_capacity)

        main_args = [args, batch_queue, device_descriptor, workshop_semaphore, ranks[process_index], logger]
        init_args = [device, master_ip, master_port, world_size, ranks[process_index]]
        consumer = torch.multiprocessing.Process(
            target=distributed_main,
            args=(
                process_main,
                main_args,
                init_args,
                distributed_manager,
            ),
            daemon=True
        )
        distributed_manager.manage(consumer)
        logger.info(f' * No.{process_index} Training Process start ...')
        logger.info(f'   PID: {consumer.pid};')
        logger.info(f'   Rank: {ranks[process_index]}/{world_size};')
        logger.info(f'   Device: {device_descriptor}.')
        consumer.start()

        consumers.append(consumer)
        batch_queues.append(batch_queue)
        device_descriptors.append(device_descriptor)

    producer = torch.multiprocessing.Process(
        target=build_batches,
        args=(
            args,
            batch_queues,
            device_descriptors,
            workshop_semaphore,
            world_size,
        ),
        daemon=True
    )
    distributed_manager.manage(producer)
    logger.info(f' * Batch Producer Process start (PID: {producer.pid}).')
    producer.start()

    for consumer in consumers:
        consumer.join()
    producer.join()

    logger.info(' = Finished !')


def main():
    args = harg.get_arguments()
    train_args = harg.get_partial_arguments(args, 'binaries.train')
    train_args.model = harg.get_partial_arguments(args, f'models.{train_args.model}')
    train(train_args)


if __name__ == '__main__':
    main()
