#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-11-17 11:08
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch


class GreedySearcher(object):
    def __init__(self,
                 search_space_size, initial_node, terminal_node,
                 min_depth=0, max_depth=512):
        self.search_space_size = search_space_size

        self.initial_node = initial_node
        self.terminal_node = terminal_node

        self.min_depth = min_depth
        self.max_depth = max_depth

        assert self.min_depth <= self.max_depth

        self.parallel_line_number = None

    def initialize(self, parallel_line_number, device_descriptor):
        self.current_depth = 0
        self.parallel_line_number = parallel_line_number

        self.paths = torch.full(
            [self.parallel_line_number, 1],
            self.initial_node,
            dtype=torch.long,
            device=device_descriptor
        )

        self.path_log_probs = torch.full(
            [self.parallel_line_number, ],
            float(0.0),
            dtype=torch.float,
            device=device_descriptor
        )

        self.results = list(dict() for i in range(self.parallel_line_number))

        self.line_original_indices = torch.arange(self.parallel_line_number, device=device_descriptor)

        self.line_finished_flags = torch.full(
            [self.parallel_line_number],
            False,
            dtype=torch.bool,
            device=device_descriptor
        )

    @property
    def current_nodes(self):
        return self.paths[:, -1]

    @property
    def found_nodes(self):
        return self.paths

    @property
    def finished(self):
        if len(self.line_original_indices) == 0:
            return True
        else:
            return False

    def update(self):
        self.line_finished_flags = self.current_nodes.eq(self.terminal_node)

        if self.current_depth == self.max_depth + 1:
            self.line_finished_flags.fill_(True)

        finished_line_indices = self.line_finished_flags.nonzero(as_tuple=False).view(-1)
        active_line_indices = (~self.line_finished_flags).nonzero(as_tuple=False).view(-1)

        for finished_line_index in finished_line_indices:
            line_original_index = self.line_original_indices[finished_line_index]
            self.results[line_original_index]['log_prob'] = self.path_log_probs[finished_line_index].item()
            self.results[line_original_index]['path'] = self.paths[finished_line_index, 1:]
            self.parallel_line_number -= 1

        self.line_original_indices = torch.index_select(self.line_original_indices, 0, active_line_indices)
        self.path_log_probs = torch.index_select(self.path_log_probs, 0, active_line_indices)
        self.paths = torch.index_select(self.paths, 0, active_line_indices)
        self.line_finished_flags = torch.index_select(self.line_finished_flags, 0, active_line_indices)

    def search(self, adjacent_node_log_probs):
        assert adjacent_node_log_probs.size(0) == self.parallel_line_number
        assert adjacent_node_log_probs.size(1) == self.search_space_size
        if self.current_depth <= self.min_depth:
            adjacent_node_log_probs[:, self.terminal_node] = float("-inf")

        best_node_log_probs, best_node_indices = torch.topk(
            adjacent_node_log_probs,
            1,
        )

        best_node_log_probs = best_node_log_probs.squeeze(-1)

        assert self.path_log_probs.size() == best_node_log_probs.size()
        self.path_log_probs = self.path_log_probs + best_node_log_probs

        self.paths = torch.cat(
            [
                self.paths,
                best_node_indices
            ],
            -1
        )

        self.current_depth += 1

        return
