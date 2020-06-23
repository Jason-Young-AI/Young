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
