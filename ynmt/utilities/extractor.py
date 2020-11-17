#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:10
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch


def get_tiled_tensor(tensor, dimension, times):
    order = [ index for index in range(tensor.dim())]
    order[0], order[dimension] = order[dimension], order[0]

    tensor = tensor.permute(order).contiguous()

    tiled_size = list(tensor.size())
    tiled_size[0] *= times

    tensor = tensor \
            .reshape(tensor.size(0), -1) \
            .transpose(0, 1) \
            .repeat(times, 1) \
            .transpose(0, 1) \
            .reshape(tiled_size)

    tensor = tensor.permute(order).contiguous()

    return tensor


def get_model_parameters_number(model):
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_foresee_mask(in_dimension, out_dimension, device, foresee_number=0):
    return torch.triu(torch.ones([in_dimension, out_dimension], dtype=torch.bool, device=device), diagonal=foresee_number+1)


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
