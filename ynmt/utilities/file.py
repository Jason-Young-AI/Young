#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-07 23:21
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import io
import os
import torch
import pickle
import tempfile


def get_temp_file_path(prefix):
    sys_temp_dir_path = tempfile.gettempdir()
    temp_dir_path = tempfile.mkdtemp(dir=sys_temp_dir_path, prefix=prefix)
    logging_file, logging_path = tempfile.mkstemp(dir=temp_dir_path)
    return logging_file, logging_path


def dumps(data):
    serailized_data = None
    bytes_storage = io.BytesIO()
    torch.save(data, bytes_storage)
    serialized_data = bytes_storage.getvalue()
    return serialized_data


def loads(serialized_data):
    data = None
    bytes_storage = io.BytesIO(serialized_data)
    data = torch.load(bytes_storage)
    return data


def safely_readline(binary_file_object):
    current_position = binary_file_object.tell()
    while True:
        try:
            line = binary_file_object.readline()
            return line.decode(encoding='utf-8')
        except UnicodeDecodeError:
            current_position -= 1
            binary_file_object.seek(current_position)


def blocks(binary_file_object, edge_start, edge_end, byte_size=1024*1024*1024):
    binary_file_object.seek(edge_start)
    current_tell = binary_file_object.tell()
    while binary_file_object.tell() < edge_end:
        block_size = min(byte_size, edge_end-binary_file_object.tell())
        yield binary_file_object.read(block_size)


def count_line(binary_file_object, edge_start, edge_end):
    number_line = 0
    for block in blocks(binary_file_object, edge_start, edge_end):
        number_line += block.count(b'\n')
    return number_line


def get_coedges(file_path, edges, cofile_path):
    coedges = list()
    cocurrent_start = 0
    cocurrent_end = 0
    with open(file_path, 'rb') as file_object, open(cofile_path, 'rb') as cofile_object:
        for edge_start, edge_end in edges:
            number_line = count_line(file_object, edge_start, edge_end)
            cocurrent_start = cocurrent_end
            while number_line:
                safely_readline(cofile_object)
                number_line -= 1
            cocurrent_end = cofile_object.tell()
            coedges.append((cocurrent_start, cocurrent_end))
    return coedges


def file_slice_edges(file_path, number_slice):
    file_size = os.path.getsize(file_path)
    quotient, remainder = divmod(file_size,  number_slice)
    edges = list()
    current_start = 0
    current_end = 0
    with open(file_path, 'rb') as file_object:
        for index in range(number_slice):
            current_start = current_end
            current_end = current_start + quotient + int( index < remainder )
            file_object.seek(current_end)
            safely_readline(file_object)
            if file_object.tell() < file_size:
                current_end = file_object.tell()
            else:
                current_end = file_size
            edges.append((current_start, current_end))
    return edges


def load_data_objects(file_path):
    with open(file_path, 'rb') as file_object:
        while True:
            try:
                yield pickle.load(file_object)
            except EOFError:
                break


def save_data_objects(file_path, data_objects):
    with open(file_path, 'wb') as file_object:
        for data_object in data_objects:
            pickle.dump(data_object, file_object)
