#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2021-03-23 06:42
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch


class CapsuleLayer(torch.nn.Module):
    def __init__(self,
        input_vector_number, input_vector_size,
        output_vector_number, output_vector_size,
        routing_type='dynamic', routing_iteration=3, share_routing_weight=True,
        squashing_epsilon=0.5
    ):
        super(CapsuleLayer, self).__init__()

        self.input_vector_number = input_vector_number
        self.input_vector_size = input_vector_size
        self.output_vector_number = output_vector_number
        self.output_vector_size = output_vector_size

        if routing_type not in {'dynamic', 'expectation_maximization', 'no'}:
            raise ValueError('No such type of routing strategy: {}'.format(routing_type))
        self.routing_type = routing_type

        if self.routing_type != 'no':
            self.routing_iteration = routing_iteration
            self.share_routing_weight = share_routing_weight

            # weight matrix:
            # [input_vector_number x output_vector_number x output_vector_size x input_vector_size]
            if self.share_routing_weight:
                self.weight_matrix = torch.nn.Parameter(
                    0.01 * torch.randn(
                        1, self.output_vector_number,
                        self.output_vector_size, self.input_vector_size,
                    )
                )
            else:
                assert self.input_vector_number >= 1, f'#1 arg must be greater than or equal to 1.'
                self.weight_matrix = torch.nn.Parameter(
                    0.01 * torch.randn(
                        self.input_vector_number, self.output_vector_number,
                        self.output_vector_size, self.input_vector_size,
                    )
                )

        self.squashing_epsilon = squashing_epsilon

    def forward(self, input_vectors):
        # input vectors:
        # [batch_size x input_vector_number x input_vector_size]
        input_vectors = self.squash(input_vectors, epsilon=self.squashing_epsilon)

        # output vectors:
        # [batch_size x output_vector_number x output_vector_size]
        output_vectors = self.route(input_vectors)

        return output_vectors

    def dynamic_routing(self, input_vectors):
        batch_size = input_vectors.size(0)
        input_vector_number = input_vectors.size(1)

        if self.share_routing_weight:
            pass
        else:
            assert input_vector_number == self.input_vector_number, f'Invalid number of input vector {input_vector_number}!'

        # prediction vectors:
        # [batch_size x input_vector_number x output_vector_number x output_vector_size]
        # =
        # weight matrix
        # [input_vector_number x output_vector_number x { output_vector_size x input_vector_size }]
        # *
        # input_vectors
        # [batch_size x input_vector_number x 1(None) x { input_vector_size x 1(None) }]
        prediction_vectors = torch.squeeze(torch.matmul(self.weight_matrix, input_vectors[:, :, None, :, None]), dim=-1)
        detached_prediction_vectors = prediction_vectors.detach()

        # logits:
        # [batch_size x input_vector_number x output_vector_number]
        logits = torch.zeros(batch_size, input_vector_number, self.output_vector_number, requires_grad=True).cuda()

        for iteration in range(self.routing_iteration):
            # coupling coefficients:
            # [batch_size x input_vector_number x output_vector_number]
            coupling_coefficients = torch.nn.functional.softmax(logits, dim=-1)

            # total input vector:
            # [batch_size x 1 x output_vector_number x output_vector_size]
            # =
            # sigma {i} ( 'where {i} is input_vector_number {j} is output_vector_number'
            #  coupling coefficients {i,j}
            #  [batch_size x input_vector_number x output_vector_number x 1(None)]
            #  *
            #  prediction vectors {i, j}
            #  [batch_size x input_vector_number x output_vector_number x output_vector_size]
            # )
            total_input_vector = torch.sum(
                coupling_coefficients[:, :, :, None] * prediction_vectors,
                dim=1,
                keepdim=True
            )

            # output vectors:
            # [batch_size x 1 x output_vector_number x output_vector_size]
            output_vectors = self.squash(total_input_vector, epsilon=self.squashing_epsilon)

            if iteration == self.routing_iteration - 1:
                pass
            else:
                # agreements:
                # [batch_size x input_vector_number x output_vector_number]
                # =
                # detached_prediction_vectors
                # [batch_size x input_vector_number x output_vector_number x output_vector_size]
                # (Dot Product)
                # output_vectors
                # [batch_size x 1 x output_vector_number x output_vector_size]
                agreements = torch.sum(
                    detached_prediction_vectors * output_vectors,
                    dim=-1,
                    keepdim=False
                )
                logits = logits + agreements

        return torch.squeeze(output_vectors, dim=1)

    def expectation_maximization_routing(self, input_vectors):
        return input_vectors

    def no_routing(self, input_vectors):
        return input_vectors

    def route(self, input_vectors):
        if self.routing_type == 'dynamic':
            routing_method = self.dynamic_routing

        if self.routing_type == 'expectation_maximization':
            routing_method = self.expectation_maximization_routing

        if self.routing_type == 'no':
            routing_method = self.no_routing

        output_vectors = routing_method(input_vectors)

        return output_vectors

    def squash(self, total_input_vector, epsilon=0.5):
        euclidean_norm = torch.linalg.norm(total_input_vector, ord=2, dim=-1, keepdim=True)
        output_vectors = (euclidean_norm**2) * total_input_vector / ((epsilon + euclidean_norm**2) * (1e-8 + euclidean_norm))
        return output_vectors
