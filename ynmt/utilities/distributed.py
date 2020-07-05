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


def gather_all(data, data_size, device_descriptor):
    data_size_limit = 256 * 256
    assert data_size < data_size_limit, 'data_size exceeds data_size_limit'

    world_size = torch.distributed.get_world_size()
    gathered_datas = list()

    distributed_tensor = torch.ByteTensor(data_size + 2, device=device_descriptor)
    gathered_tensors = [torch.ByteTensor(data_size + 2, device=device_descriptor) for _ in range(world_size)]

    pickled_data = pickle.dumps(data)
    pickled_data_size = len(pickled_data)
    assert pickled_data_size <= data_size, f'data size exceeds data_size: {data_size}'

    quotient, remainder = divmod(pickled_data_size,  256)
    distributed_tensor[0] = quotient
    distributed_tensor[1] = remainder
    distributed_tensor[2:2+pickled_data_size] = torch.ByteTensor(list(pickled_data), device=device_descriptor)

    torch.distributed.all_gather(gathered_tensors, distributed_tensor)

    for gathered_tensor in gathered_tensors:
        quotient = gathered_tensor[0].item()
        remainder = gathered_tensor[1].item()
        pickled_data_size = quotient * 256 + remainder
        pickled_data = bytes(gathered_tensor[2:2+pickled_data_size].tolist())
        gathered_data = pickled.loads(pickled_data)
        gathered_datas.append(gathered_data)

    return gathered_datas


def distributed_init(device, master_ip, master_port, world_size, rank)
    backends = {'GPU': torch.distributed.Backend.NCCL, 'CPU': torch.distributed.Backend.GLOO}
    init_method = f'tcp://{master_ip}:{master_port}'
    torch.distributed.init_process_group(
        backends[device],
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )


def distributed_main(main, main_args, init_args, distributed_manager):
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
        distributed_manager.send_message(os.getpid(), exception)
        sys.exit(1)


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

    def send_message(self, pid, exception):
        message = (pid, exception)
        self.exception_queue.put(message)

    def exception_catcher(self):
        pid, exception = self.exception_queue.get()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()

        exception_message = f'''
            \n
            -- Tracebacks above this line can probably be ignored --
            Process {pid} terminated with the following error:
            {exception}
            \n
        '''
        raise Exception(exception_message)
