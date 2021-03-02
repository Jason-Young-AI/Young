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


import torch

from yoolkit.logging import setup_logger, logging_level
from yoolkit.visualizing import setup_visualizer
from yoolkit.registration import import_modules

import ynmt.hocon.arguments as harg

from ynmt.utilities.apex import get_apex, mix_precision
from ynmt.utilities.random import fix_random_procedure
from ynmt.utilities.checkpoint import load_checkpoint
from ynmt.utilities.extractor import get_model_parameters_number
from ynmt.utilities.distributed import DistributedManager, distributed_main, distributed_data_sender, distributed_data_receiver, get_device_descriptor

from ynmt.factories import build_factory
from ynmt.models import build_model, model_registration
from ynmt.schedulers import build_scheduler
from ynmt.optimizers import build_optimizer
from ynmt.trainers import build_trainer
from ynmt.testers import build_tester


def process_main(args, batch_queue, device_descriptor, workshop_semaphore, rank):
    import_modules(args.user_defined_modules_directory, 'ynmt.user_defined')
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)
    fix_random_procedure(args.random_seed)

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

    testing_batches = distributed_data_receiver(batch_queue, workshop_semaphore, 'small')
    validation_batches = distributed_data_receiver(batch_queue, workshop_semaphore, 'small')
    training_batches = distributed_data_receiver(batch_queue, workshop_semaphore, 'large')

    ## Building Something
    
    # Build factory
    logger.info(f' + Building Factory: [\'{args.factory.name}\'] ...')
    factory = build_factory(args.factory, logger)
    logger.info(f'   The construction of Factory [\'{factory.__class__.__name__}\'] is complete.')

    # Load Ancillary Datasets
    logger.info(f' + Loading Ancillary Datasets ...')
    factory.load_ancillary_datasets(args.factory.args)
    logger.info(f'   Ancillary Datasets has been loaded.')

    # Build Model
    logger.info(f' + Building Model: [\'{args.model.name}\'] ...')
    model = build_model(args.model, factory)
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

    logger.info(f' + Moving model to the specified device ...')
    model.to(device_descriptor)
    logger.info(f'   Complete.')

    # Build Scheduler
    logger.info(f' + Building Scheduler [\'{args.scheduler.name}\'] ...')
    scheduler = build_scheduler(args.scheduler, model)
    logger.info(f'   The construction of Scheduler [\'{scheduler.__class__.__name__}\'] is complete.')

    # Build Optimizer
    logger.info(f' + Building Optimizer [\'{args.optimizer.name}\'] ...')
    optimizer = build_optimizer(args.optimizer, model)
    logger.info(f'   The construction of Optimizer [\'{optimizer.__class__.__name__}\'] is complete.')

    model, optimizer = mix_precision(model, optimizer, mix_precision=args.mix_precision.on, optimization_level=args.mix_precision.optimization_level)
    if args.mix_precision.on:
        apex = get_apex()
        if apex is not None:
            logger.info(f' ! Training in mix precision mode, optimization level is {args.mix_precision.optimization_level}')
        else:
            logger.info(f' ! Not found NVIDIA-APEX module! Now training in normal mode.')

    # Loading Checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint is None:
        logger.info(f' ~ Training from scratch.')
    else:
        logger.info(f' ~ Training from checkpoint [\'{args.checkpoint}\'] at [{checkpoint["step"]}] steps.')

        if args.reset_scheduler:
            logger.info(f'   Reset Scheduler.')
        else:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        if args.reset_optimizer:
            logger.info(f'   Reset Optimizer.')
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        logger.info(f'   Personalized Loading Parameters ...')
        model.personalized_load_state(checkpoint['model_state'])
        logger.info(f'   Loaded.')

    # Build Tester
    logger.info(f' + Building Tester [\'{args.tester.name}\'] ...')
    tester = build_tester(args.tester, factory, model, device_descriptor, logger)
    logger.info(f'   The construction of Tester [\'{tester.__class__.__name__}\'] is complete.')

    # Build Trainer
    logger.info(f' + Building Trainer [\'{args.trainer.name}\'] ...')
    trainer = build_trainer(args.trainer, factory, model, scheduler, optimizer, tester, device_descriptor, logger, visualizer)

    if checkpoint is not None:
        if args.reset_trainer:
            logger.info(f'   Reset Step.')
        else:
            trainer.step = checkpoint['step']

    logger.info(f'   The construction of Trainer [\'{trainer.__class__.__name__}\'] is complete.')

    if visualizer.disabled:
        logger.info(f' * Visualizer Disabled.')
    else:
        # Open Visualizer
        logger.info(f' * Visualizer On.')
        visualizer.open()
        if visualizer.offline:
            logger.info(f'   Visualizer in Offline mode')
        else:
            logger.info(f'   Visualizer connection established between {visualizer.server}:{visualizer.port}')
        logger.info(f'   Visualizer logging to [\'{visualizer.logging_path}\'] .')

    # Launch Trainer
    logger.info(f' * Launch Trainer ...')
    logger.info(f'   Trainer Life Cycle: {trainer.life_cycle} update steps!')
    logger.info(f'   Saving checkpoint every {trainer.training_period} steps;')
    logger.info(f'   Validate every {trainer.validation_period} steps;')
    logger.info(f'   Test every {trainer.testing_period} steps.')

    trainer.launch(training_batches, validation_batches, testing_batches)

    visualizer.close()


def build_batches(args, batch_queues, workshop_semaphore, world_size, ranks):
    import_modules(args.user_defined_modules_directory, 'ynmt.user_defined')
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)
    logger.disabled = True

    fix_random_procedure(args.random_seed)

    factory = build_factory(args.factory, logger)
    factory.load_ancillary_datasets(args.factory.args)

    distributed_data_sender(factory.testing_batches(args.factory.args), batch_queues, workshop_semaphore, world_size, ranks)
    distributed_data_sender(factory.validation_batches(args.factory.args), batch_queues, workshop_semaphore, world_size, ranks)
    distributed_data_sender(factory.training_batches(args.factory.args), batch_queues, workshop_semaphore, world_size, ranks)


def train(args):
    logger = setup_logger(args.logger.name, logging_path=args.logger.path, logging_level=logging_level['INFO'], to_console=args.logger.console_report)

    logger.info(
        f'\n The following is the arguments:'
        f'\n{args}'
    )

    device = args.distribution.device
    master_ip = args.distribution.master_ip
    master_port = args.distribution.master_port
    world_size = args.distribution.world_size
    ranks = args.distribution.ranks
    workshop_capacity = args.distribution.workshop_capacity
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
    producer.terminate()

    logger.info(' $ Finished !')


def main():
    train_args = harg.get_arguments('train')
    train(train_args)


if __name__ == '__main__':
    main()
