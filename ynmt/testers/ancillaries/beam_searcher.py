#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-05 18:39
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


class BeamSearcher(object):
    def __init__(self,
                 reserved_path_number, candidate_path_number, search_space_size, initial_node, terminal_node,
                 min_depth=0, max_depth=512, alpha=0.0, beta=0.0):
        self.reserved_path_number = reserved_path_number
        self.candidate_path_number = candidate_path_number
        self.search_space_size = search_space_size

        self.initial_node = initial_node
        self.terminal_node = terminal_node

        self.min_depth = min_depth
        self.max_depth = max_depth

        assert self.min_depth <= self.max_depth

        self.alpha = alpha
        self.beta = beta

        self.parallel_line_number = None

    def initialize(self, parallel_line_number, device_descriptor):
        self.current_depth = 0
        self.parallel_line_number = parallel_line_number

        self.reserved_paths = torch.full(
            [self.parallel_line_number, self.reserved_path_number, 1],
            self.initial_node,
            dtype=torch.long,
            device=device_descriptor
        )

        self.reserved_path_log_probs = torch.full(
            [self.parallel_line_number, self.reserved_path_number],
            float("-inf"),
            dtype=torch.float,
            device=device_descriptor
        )
        self.reserved_path_log_probs[:, 0] = 0

        self.reserved_path_scores = torch.full(
            [self.parallel_line_number, self.reserved_path_number],
            float("-inf"),
            dtype=torch.float,
            device=device_descriptor
        )
        self.reserved_path_scores[:, 0] = 0

        self.candidate_paths = list(
            list()
            for _ in range(self.parallel_line_number)
        )

        self.active_line_indices = torch.arange(self.parallel_line_number, device=device_descriptor)
        self.line_order = torch.arange(self.parallel_line_number, device=device_descriptor)
        self.finished_lines = torch.full(
            [self.parallel_line_number],
            False,
            dtype=torch.bool,
            device=device_descriptor
        )
        self.path_offset = torch.arange(
            0,
            self.parallel_line_number * self.reserved_path_number,
            self.reserved_path_number,
            device=device_descriptor
        ).unsqueeze(-1).repeat(1, self.reserved_path_number)

    @property
    def current_nodes(self):
        return self.reserved_paths[:, :, -1]

    @property
    def found_nodes(self):
        return self.reserved_paths

    @property
    def finished(self):
        finished_paths = self.current_nodes.eq(self.terminal_node)

        if self.current_depth == self.max_depth + 1:
            finished_paths.fill_(True)

        self.finished_lines |= finished_paths[:, 0].eq(True)

        new_active_line_indices = []
        for line_index in range(finished_paths.size(0)):
            active_line_index = self.active_line_indices[line_index]
            finished_path = finished_paths[line_index].nonzero().view(-1)
            for path_index in finished_path:
                self.candidate_paths[active_line_index].append(
                    dict(
                        log_prob = self.reserved_path_log_probs[line_index, path_index].item(),
                        score = self.reserved_path_scores[line_index, path_index].item(),
                        path = self.reserved_paths[line_index, path_index, 1:]
                    )
                )
            if self.finished_lines[line_index] and len(self.candidate_paths[active_line_index]) >= self.candidate_path_number:
                self.candidate_paths[active_line_index] = sorted(self.candidate_paths[active_line_index], key=lambda x: x['score'], reverse=True)
                self.candidate_paths[active_line_index] = self.candidate_paths[active_line_index][:self.candidate_path_number]
                self.parallel_line_number -= 1
            else:
                new_active_line_indices.append(line_index)

        new_active_line_indices = torch.tensor(
            new_active_line_indices,
            dtype=torch.long,
            device=self.active_line_indices.device
        )

        self.active_line_indices = torch.index_select(self.active_line_indices, 0, new_active_line_indices)
        self.reserved_path_log_probs = torch.index_select(self.reserved_path_log_probs, 0, new_active_line_indices)
        self.reserved_path_scores = torch.index_select(self.reserved_path_scores, 0, new_active_line_indices)
        self.reserved_paths = torch.index_select(self.reserved_paths, 0, new_active_line_indices)
        self.finished_lines = torch.index_select(self.finished_lines, 0, new_active_line_indices)
        self.path_offset = torch.index_select(self.path_offset, 0, new_active_line_indices)

        if len(new_active_line_indices) == 0:
            return True
        else:
            return False

    def search(self, adjacent_node_log_probs):
        assert adjacent_node_log_probs.size(0) == self.parallel_line_number
        assert adjacent_node_log_probs.size(1) == self.reserved_path_number
        assert adjacent_node_log_probs.size(2) == self.search_space_size
        if self.current_depth <= self.min_depth:
            adjacent_node_log_probs[:, :, self.terminal_node] = float("-inf")

        adjacent_node_log_probs = adjacent_node_log_probs.reshape(-1, self.search_space_size)

        extended_path_log_probs = self.reserved_path_log_probs.reshape(-1, 1) + adjacent_node_log_probs
        extended_path_log_probs = extended_path_log_probs.reshape(self.parallel_line_number, -1)

        length_normalization = ((5 + self.current_depth)/(5 + 1)) ** self.alpha
        coverage_penalty = self.beta

        extended_path_scores = extended_path_log_probs / length_normalization + coverage_penalty

        topk_extended_path_scores, topk_extended_path_indices = torch.topk(
            extended_path_scores,
            self.reserved_path_number,
        )

        assert self.reserved_path_scores.size() == topk_extended_path_scores.size()
        self.reserved_path_scores = topk_extended_path_scores

        # update reserved_path_log_probs
        line_offset = topk_extended_path_indices + \
                      self.line_order[:self.parallel_line_number].unsqueeze(1) * \
                      self.reserved_path_number * \
                      self.search_space_size

        self.reserved_path_log_probs = torch.index_select(
            extended_path_log_probs.reshape(-1),
            0,
            line_offset.reshape(-1)
        ).reshape(self.parallel_line_number, self.reserved_path_number)

        # update reserved_paths
        adjacent_nodes = torch.fmod(topk_extended_path_indices, self.search_space_size)

        self.path_offset = torch.floor_divide(topk_extended_path_indices, self.search_space_size) + \
                      self.line_order[:self.parallel_line_number].unsqueeze(1) * \
                      self.reserved_path_number

        self.reserved_paths = torch.index_select(
            self.reserved_paths.reshape(self.parallel_line_number * self.reserved_path_number, -1),
            0,
            self.path_offset.reshape(-1)
        ).reshape(self.parallel_line_number, self.reserved_path_number, -1)

        self.reserved_paths = torch.cat(
            [
                self.reserved_paths,
                adjacent_nodes.unsqueeze(-1)
            ],
            -1
        )

        self.current_depth += 1

        return
