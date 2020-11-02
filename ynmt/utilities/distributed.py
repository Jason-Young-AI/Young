#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-18 11:52
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import threading

from yoolkit.gio import dumps, loads


def get_device_descriptor(device, index):
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{index}'

    return torch.device(device_name)


def distributed_data_sender(data_generator, data_queues, workshop_semaphore, world_size, ranks, order_index=False):
    rank2index = dict()
    for index, rank in enumerate(ranks):
        rank2index[rank] = index

    for index, data in enumerate(data_generator):
            rank = index % world_size
            if rank not in set(ranks):
                continue
            else:
                workshop_semaphore.acquire()
                if order_index:
                    data = (index, data)
                data_queues[rank2index[rank]].put(data)

    for data_queue in data_queues:
        data_queue.put(None)


def distributed_data_receiver(receiver_type, data_queue, workshop_semaphore):
    assert receiver_type in {'list', 'generator'}
    if receiver_type == 'list':
        return distributed_data_receive_as_list(data_queue, workshop_semaphore)
    if receiver_type == 'generator':
        return distributed_data_receive_as_generator(data_queue, workshop_semaphore)


def distributed_data_receive_as_list(data_queue, workshop_semaphore):
    data_list = list()
    while True:
        data = data_queue.get()
        workshop_semaphore.release()
        if data is None:
            return data_list
        else:
            data_list.append(data)


def distributed_data_receive_as_generator(data_queue, workshop_semaphore):
    while True:
        data = data_queue.get()
        workshop_semaphore.release()
        if data is None:
            return
        else:
            yield data


def gather_all(data, device_descriptor, data_size=8 * 1024):
    data_size_limit = 256 * 256
    assert data_size < data_size_limit, 'Error: data_size exceeds data_size_limit'

    world_size = torch.distributed.get_world_size()
    gathered_datas = list()

    distributed_tensor = torch.zeros(data_size + 2, dtype=torch.uint8, device=device_descriptor)
    gathered_tensors = [torch.zeros(data_size + 2, dtype=torch.uint8, device=device_descriptor) for _ in range(world_size)]


    serialized_data = dumps(data)
    serialized_data_size = len(serialized_data)
    assert serialized_data_size <= data_size, f'Data size exceeds data_size: {data_size}'

    quotient, remainder = divmod(serialized_data_size, 256)
    distributed_tensor[0] = quotient
    distributed_tensor[1] = remainder
    distributed_tensor[2:2+serialized_data_size] = torch.tensor(list(serialized_data), dtype=torch.uint8, device=device_descriptor)

    torch.distributed.all_gather(gathered_tensors, distributed_tensor)

    for gathered_tensor in gathered_tensors:
        quotient = gathered_tensor[0].item()
        remainder = gathered_tensor[1].item()
        serialized_data_size = quotient * 256 + remainder
        serialized_data = bytes(gathered_tensor[2:2+serialized_data_size].tolist())
        gathered_data = loads(serialized_data)
        gathered_datas.append(gathered_data)

    return gathered_datas


def reduce_all(tensors, device_descriptor, data_size=8 * 1024 * 1024):
    data_size_limit = 128 * 1024 * 1024
    assert data_size < data_size_limit, 'Error: data_size exceeds data_size_limit'

    tensor_type = tensors[0].dtype
    for tensor in tensors:
        assert tensor.dtype == tensor_type, f'All dtype of tensor in tensor has to be the same({tensor_type}).'

    data = torch.zeros(data_size, dtype=tensor_type, device=device_descriptor)

    cache = list()
    cache_size = 0

    def reduce_all_data():
        index = 0
        for tensor in cache:
            tensor_size = tensor.numel()
            data[index:index+tensor_size].copy_(tensor.reshape(-1))
            index += tensor_size

        torch.distributed.all_reduce(data[:index])

        index = 0
        for tensor in cache:
            tensor_size = tensor.numel()
            tensor.reshape(-1).copy_(data[index:index+tensor_size])
            index += tensor_size

    for tensor in tensors:
        tensor_size = tensor.numel()
        if tensor_size > data_size:
            torch.distributed.all_reduce(tensor)
            continue
        if cache_size + tensor_size > data_size:
            reduce_all_data()
            cache = list()
            cache_size = 0
        cache.append(tensor)
        cache_size += tensor_size

    if len(cache) != 0:
        reduce_all_data()
        cache = list()
        cache_size = 0


def distributed_init(device, master_ip, master_port, world_size, rank):
    backends = {'GPU': torch.distributed.Backend.NCCL, 'CPU': torch.distributed.Backend.GLOO}
    init_method = f'tcp://{master_ip}:{master_port}'
    torch.distributed.init_process_group(
        backends[device],
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )


def distributed_main(main, main_args, init_args, exception_queue):
    try:
        distributed_init(*init_args)
        main(*main_args)
    except KeyboardInterrupt:
        pass
    except Exception:
        import os
        import sys
        import traceback
        exception = "".join(traceback.format_exception(*sys.exc_info()))
        message = (os.getpid(), exception)
        exception_queue.put(message)


class DistributedManager(object):
    def __init__(self):
        self.exception_queue = torch.multiprocessing.Queue()
        self.processes = list()
        self.exception_catcher_thread = threading.Thread(
            target=self.exception_catcher,
            args=(),
            daemon=True
        )
        self.exception_catcher_thread.start()

    def manage(self, process):
        self.processes.append(process)

    def open(self):
        self.exception_catcher_thread.start()

    def exception_catcher(self):
        pid, exception = self.exception_queue.get()

        exception_message = (
            f'\n'
            f'\n\t\t -- Tracebacks above this line can probably be ignored -- '
            f'\n\t\t    Process {pid} terminated with the following error:'
            f'\n'
            f'\n{exception}'
        )
        raise Exception(exception_message)
