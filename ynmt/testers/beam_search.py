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


def build_tester_beam_search(args, vocabulary):
    beam_search = BeamSearch(
        reserved_path_number = args.beam_size,
        candidate_path_number = args.n_best,
        search_space_size = len(vocabulary),
        initial_node = vocabulary.bos_index,
        terminal_node = vocabulary.eos_index,
        min_depth = args.min_length, max_depth = args.max_length,
        alpha = args.penalty.alpha, beta = args.penalty.beta
    )
    return beam_search


class BeamSearch(object):
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
        self.adjacent_node_getter = None

    def initialize(self, parallel_line_number, adjacent_node_states_getter):
        self.current_depth = 0
        self.parallel_line_number = parallel_line_number
        self.adjacent_node_states_getter = adjacent_node_states_getter

        self.reserved_pathes = torch.full(
            [self.parallel_line_number, self.reserved_path_number, 1],
            self.initial_node,
            dtype=torch.long
        )

        self.reserved_path_log_probs = torch.full(
            [self.parallel_line_number, self.reserved_path_number],
            float("-inf"),
            dtype=torch.float
        )
        self.reserved_path_log_probs[:, 0] = 0

        self.reserved_path_scores = torch.full(
            [self.parallel_line_number, self.reserved_path_number],
            float("-inf"),
            dtype=torch.float
        )
        self.reserved_path_scores[:, 0] = 0

        self.candidate_pathes = list(
            list()
            for _ in range(self.parallel_line_number)
        )

        self.active_line_indices = torch.arange(self.parallel_line_number)
        self.line_order = torch.arange(self.parallel_line_number)
        self.finished_lines = torch.full([self.parallel_line_number], False, dtype=torch.bool)

    @property
    def finished(self):
        finished_pathes = self.reserved_pathes[:, :, -1].eq(self.terminal_node)
        if self.current_depth == self.max_depth:
            finished_pathes.fill_(True)

        self.finished_lines = self.finished_lines | finished_pathes[:, 0].eq(True)
        new_active_line_indices = []
        for line_index, finished_path in enumerate(finished_pathes):
            active_line_index = self.active_line_indices[line_index]
            for path_index in finished_path.nonzero():
                print(finished_pathes.size())
                print(line_index)
                print(active_line_index)
                self.candidate_pathes[active_line_index].append(
                    dict(
                        prob = self.reserved_path_log_probs[line_index][path_index],
                        score = self.reserved_path_scores[line_index][path_index],
                        path = self.reserved_pathes[line_index][path_index]
                    )
                )
            if len(self.candidate_pathes[active_line_index]) >= self.candidate_path_number and self.finished_lines[line_index]:
                self.candidate_pathes[active_line_index] = sorted(self.candidate_pathes[active_line_index], key=lambda x: x['score'], reverse=True)
                self.candidate_pathes[active_line_index] = self.candidate_pathes[active_line_index][:self.candidate_path_number]
                self.parallel_line_number -= 1
            else:
                new_active_line_indices.append(line_index)

        new_active_line_indices = torch.tensor(new_active_line_indices, dtype=torch.long)

        self.active_line_indices = torch.index_select(self.active_line_indices, 0, new_active_line_indices)
        self.reserved_path_log_probs = torch.index_select(self.reserved_path_log_probs, 0, new_active_line_indices)
        self.reserved_path_scores = torch.index_select(self.reserved_path_scores, 0, new_active_line_indices)
        self.reserved_pathes = torch.index_select(self.reserved_pathes, 0, new_active_line_indices)
        self.finished_lines = torch.index_select(self.finished_lines, 0, new_active_line_indices)

        if len(new_active_line_indices) == 0:
            return True
        else:
            return False

    def get_adjacent_node_states(self, nodes):
        adjacent_node_states = self.adjacent_node_states_getter(nodes)

        adjacent_node_probs = adjacent_node_states['prob']
        assert adjacent_node_probs.size(0) == self.parallel_line_number
        assert adjacent_node_probs.size(1) == self.reserved_path_number
        assert adjacent_node_probs.size(2) == self.search_space_size

        if self.current_depth < self.min_depth:
            adjacent_node_probs[:, :, self.terminal_node] = 0

        adjacent_node_states = dict(
            prob = adjacent_node_probs
        )
        return adjacent_node_states

    def search(self):
        while not self.finished:
            nodes = self.reserved_pathes[:, :, -1]
            adjacent_node_states = self.get_adjacent_node_states(nodes)
            adjacent_node_probs = adjacent_node_states['prob']
            adjacent_node_log_probs = torch.log(adjacent_node_probs).reshape(-1, self.search_space_size)

            extended_path_log_probs = self.reserved_path_log_probs.reshape(-1, 1) + adjacent_node_log_probs
            extended_path_log_probs = extended_path_log_probs.reshape(self.parallel_line_number, -1)

            length_normalization = ((5 + self.current_depth)/(5 + 1)) ** self.alpha
            coverage_penalty = self.beta

            extended_path_scores = extended_path_log_probs / length_normalization + coverage_penalty

            topk_extended_path_scores, topk_extended_path_indices = torch.topk(
                extended_path_scores,
                self.reserved_path_number,
            )

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

            # update reserved_pathes
            adjacent_nodes = torch.fmod(topk_extended_path_indices, self.search_space_size)

            path_offset = torch.floor_divide(topk_extended_path_indices, self.search_space_size) + \
                          self.line_order[:self.parallel_line_number].unsqueeze(1) * \
                          self.reserved_path_number

            self.reserved_pathes = torch.index_select(
                self.reserved_pathes.reshape(self.parallel_line_number * self.reserved_path_number, -1),
                0,
                path_offset.reshape(-1)
            ).reshape(self.parallel_line_number, self.reserved_path_number, -1)
            self.reserved_pathes = torch.cat(
                [
                    self.reserved_pathes,
                    adjacent_nodes.unsqueeze(-1)
                ],
                -1
            )

            self.current_depth += 1

        return
