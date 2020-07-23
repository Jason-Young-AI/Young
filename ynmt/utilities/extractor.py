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


def get_match_item(x, pattern, invalid_item):
    valid_mask = pattern.ne(invalid_item)
    match_item = x.eq(pattern).masked_select(valid_mask)
    return match_item


def get_model_parameters_number(model):
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_position(tensor):
    size = list(tensor.size())
    max_position = size[-1]
    size[-1] = 1
    repeat_size = tuple(size)
    position = torch.arange(0, max_position, device=tensor.device)
    position = position.repeat(repeat_size)
    return position


def get_attend_mask(attend_emission_length, attend_scope):
    # attend_scope.size() == [batch_size x attend_scope_length]
    position = torch.arange(0, attend_scope.size(1), device=attend_scope.device).repeat(attend_scope.size(0), attend_emission_length, 1)
    mask = torch.ge(position, attend_scope.unsqueeze(1))
    return mask


def count_correct_element_number(tensor, reference_tensor):
    correct_flag = torch.eq(tensor, reference_tensor)
    correct_element = torch.sum(correct_flag)
    return correct_element.item()


def count_total_element_number(tensor, ignore_index):
    valid_flag = torch.ne(tensor, ignore_index)
    valid_element = torch.sum(valid_flag)
    return valid_element.item()
