#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-06-28 18:02
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


def get_model_parameters_number(model):
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_future_mask(tensor):
    dimension = tensor.size(-1)
    return torch.triu(torch.ones([dimension, dimension], dtype=torch.bool, device=tensor.device), diagonal=1)


def get_padding_mask(tensor, padding_index):
    return tensor == padding_index


def count_correct_element_number(tensor, reference_tensor):
    correct_flag = torch.eq(tensor, reference_tensor)
    correct_element = torch.sum(correct_flag)
    return correct_element.item()


def count_total_element_number(tensor, ignore_index):
    valid_flag = torch.ne(tensor, ignore_index)
    valid_element = torch.sum(valid_flag)
    return valid_element.item()
