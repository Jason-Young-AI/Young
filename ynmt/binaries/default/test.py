#!/usr/bin/env python3 -u
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

from yoolkit.logging import setup_logger, logging_level

import ynmt.hocon.arguments as harg

from ynmt.utilities.checkpoint import find_all_checkpoints, load_checkpoint
from ynmt.utilities.distributed import DistributedManager, distributed_main, distributed_data_sender, distributed_data_receiver, get_device_descriptor

from ynmt.factories import build_factory
from ynmt.models import build_model
from ynmt.testers import build_tester


def process_main(args, checkpoint_path, batch_queue, device_descriptor, workshop_semaphore, rank):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)
    is_station = rank == 0
    if is_station:
        logger.disabled = False | args.logger.off
    else:
        logger.disabled = True

    testing_batches = distributed_data_receiver('list', batch_queue, workshop_semaphore)

    ## Building Something

    # Build Factory
    logger.info(f' * Building factory: \'{args.factory.name}\' ...')
    factory = build_factory(args.factory, logger)
    logger.info(f'   The construction of factory is complete.')

    # Load Ancillary Datasets
    logger.info(f' * Loading Ancillary Datasets ...')
    factory.load_ancillary_datasets(args.factory.args)
    logger.info(f'   Ancillary Datasets has been loaded.')

    # Load Checkpoint & Build Model
    checkpoint = load_checkpoint(checkpoint_path)
    logger.info(f' * Checkpoint has been loaded from \'{checkpoint_path}\'')
    model_settings = checkpoint["model_settings"]
    logger.info(f' * Building Model \'{model_settings.name}\' ...')
    model = build_model(model_settings, factory)

    logger.info(f'   Loading Parameters ...')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    logger.info(f'   Loaded.')

    logger.info(f' * Moving model to device of Tester ...')
    model.to(device_descriptor)
    logger.info(f'   Complete.')

    # Build Tester
    logger.info(f' * Building Tester \'{args.tester.name}\' ...')
    tester = build_tester(args.tester, factory, model, device_descriptor, logger)
    logger.info(f'   The construction of tester is complete.')

    # Launch Tester
    logger.info(f' * Launch Tester ...')
    step = checkpoint['step']
    tester.initialize(f'step_{step}')
    tester.launch(testing_batches)


def build_batches(args, batch_queues, workshop_semaphore, world_size, ranks):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)
    logger.disabled = True

    factory = build_factory(args.factory, logger)
    factory.load_ancillary_datasets(args.factory.args)

    distributed_data_sender(factory.testing_batches(args.factory.args), batch_queues, workshop_semaphore, world_size, ranks, order_index=True)


def test(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)

    # Find checkpoints
    assert os.path.isdir(args.checkpoints.directory), f' Checkpoint directory \'{args.checkpoints.directory}\' does not exist!'

    checkpoints = find_all_checkpoints(args.checkpoints.directory, args.checkpoints.name)

    checkpoint_names = list()
    checkpoint_paths = list()
    for step, checkpoint_path in checkpoints.items():
        checkpoint_name = os.path.split(checkpoint_path)[1]
        checkpoint_names.append(checkpoint_name)
        checkpoint_paths.append(checkpoint_path)

    logger.info(f'   There are {len(checkpoints)} checkpoints will be loaded: {checkpoint_names}')

    # Distribution Testing
    device = args.distribution.device
    master_ip = "127.0.0.1"
    master_port = args.distribution.port
    workshop_capacity = args.distribution.workshop_capacity
    number_process = args.distribution.number_process
    world_size = number_process
    ranks = [i for i in range(number_process)]

    if device == 'CPU':
        logger.info(' * Distribution on CPU ...')
    if device == 'GPU':
        logger.info(' * Distribution on GPU ...')
        assert torch.cuda.device_count() >= number_process, f'Insufficient GPU!'
    logger.info(f'   Single Machine - tcp://{master_ip}:{master_port}')
    logger.info(f'   Ranks({ranks}) in World({world_size})')

    torch.multiprocessing.set_start_method('spawn')
    distributed_manager = DistributedManager()
    workshop_semaphore = torch.multiprocessing.Semaphore(world_size * workshop_capacity)

    for checkpoint_name, checkpoint_path in zip(checkpoint_names, checkpoint_paths):
        logger.info(f' * Now test checkpoint \'{checkpoint_name}\'')

        consumers = list()
        batch_queues = list()
        for process_index in range(number_process):
            device_descriptor = get_device_descriptor(device, process_index)
            batch_queue = torch.multiprocessing.Queue(workshop_capacity)

            main_args = [args, checkpoint_path, batch_queue, device_descriptor, workshop_semaphore, ranks[process_index]]
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
            logger.info(f' * No.{process_index} Testing Process start ...')
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

        logger.info(f' ! Finished testing checkpoint \'{checkpoint_name}\'')
    logger.info(f' $ Finished !')


def main():
    test_args = harg.get_arguments('test')
    test(test_args)


if __name__ == '__main__':
    main()
