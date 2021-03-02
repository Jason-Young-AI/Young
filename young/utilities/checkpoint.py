#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:09
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import torch


def find_all_checkpoints(checkpoint_directory, name):
    checkpoint_filename_pattern = re.compile(f'{name}_step_(\d+)\.cp')
    checkpoints = dict()
    for content_name in os.listdir(checkpoint_directory):
        content_path = os.path.join(checkpoint_directory, content_name)
        if os.path.isfile(content_path):
            result = checkpoint_filename_pattern.fullmatch(content_name)
            if result is not None:
                step = int(result.group(1))
                checkpoints[step] = content_path
            else:
                continue
        else:
            continue

    return checkpoints


def load_checkpoint(checkpoint_directory_or_path, name=""):
    if os.path.isfile(checkpoint_directory_or_path):
        checkpoint = torch.load(checkpoint_directory_or_path, map_location=torch.device('cpu'))

        return checkpoint
    elif os.path.isdir(checkpoint_directory_or_path):
        assert len(name) != 0, f'Invalid checkpoint name \'{name}\'.'
        checkpoints = find_all_checkpoints(checkpoint_directory_or_path, name)

        if len(checkpoints) == 0:
            latest_checkpoint = None
        else:
            max_step = max(checkpoints.keys())
            latest_checkpoint_path = checkpoints[max_step]
            if os.path.isfile(latest_checkpoint_path):
                latest_checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'))
                assert max_step == latest_checkpoint['step'], 'An Error occurred when loading checkpoint.'
            else:
                latest_checkpoint = None

        return latest_checkpoint
    else:
        return None


def save_checkpoint(checkpoint, checkpoint_directory_or_path, name="", keep_number=0):
    if os.path.isfile(checkpoint_directory_or_path):
        torch.save(checkpoint, checkpoint_directory_or_path)
    elif os.path.isdir(checkpoint_directory_or_path):
        assert len(name) != 0, f'Invalid checkpoint name \'{name}\'.'
        step = checkpoint['step']
        checkpoint_filename = f'{name}_step_{step}.cp'
        checkpoint_path = os.path.join(checkpoint_directory_or_path, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)

        checkpoints = find_all_checkpoints(checkpoint_directory_or_path, name)
        steps = sorted(list(checkpoints.keys()), reverse=True)
        for step in steps[keep_number:]:
            remove_checkpoint(checkpoints[step])
    else:
        raise IOError(f'Invalid saved address: {checkpoint_directory_or_path}')


def remove_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    else:
        raise IOError(f'Invalid address: {checkpoint_path}')
