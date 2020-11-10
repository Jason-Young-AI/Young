#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-05 20:20
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import numbers

from yoolkit.statistics import Statistics


def merge_dict(left_dict, right_dict, restrict):
    union = getattr(set, 'union')
    intersection = getattr(set, 'intersection')
    restrict_option = dict(
        max_uni = (max, union), # maximize_union
        max_its = (max, intersection), # maximize_intersect
        min_uni = (min, union), # minimize_union
        min_its = (min, intersection), # minimize_intersect
    )
    assert restrict in restrict_option, f'{restrict} must be in {restrict_option}'

    compare_method = restrict_option[restrict][0]
    set_method = restrict_option[restrict][1]

    result_keys = set_method(set(left_dict.keys()), set(right_dict.keys()))

    result_dict = dict()
    for result_key in result_keys:
        if result_key in left_dict and result_key in right_dict:
            result_dict[result_key] = compare_method(left_dict[result_key], right_dict[result_key])
        elif result_key in left_dict:
            result_dict[result_key] = left_dict[result_key]
        elif result_key in right_dict:
            result_dict[result_key] = right_dict[result_key]

    return result_dict


def perplexity(per_prediction_cross_entropy):
    per_prediction_cross_entropy = min(per_prediction_cross_entropy, 512)
    return math.exp(per_prediction_cross_entropy)


class BLEUScorer(object):
    def __init__(self, gram_number=4):
        self.gram_number = gram_number
        self.initialize()

    def __add__(self, other):
        if isinstance(other, BLEUScorer):
            assert self.gram_number == other.gram_number, f'Gram Number must be the same.'
            result_bleu_scorer = BLEUScorer(self.gram_number)
            result_bleu_scorer.total_hypothesis_length = self.total_hypothesis_length + other.total_hypothesis_length
            result_bleu_scorer.total_closest_reference_length = self.total_closest_reference_length + other.total_closest_reference_length
            for index in range(result_bleu_scorer.gram_number):
                result_bleu_scorer.ngram_statistics[index] = self.ngram_statistics[index] + other.ngram_statistics[index]
        return result_bleu_scorer

    def initialize(self):
        self.total_hypothesis_length = 0
        self.total_closest_reference_length = 0
        self.ngram_statistics = list()
        for index in range(self.gram_number):
            self.ngram_statistics.append(Statistics(set(['correct_ngram', 'total_ngram'])))

    @property
    def brevity_penalty(self):
        return min(1, math.exp(1 - self.total_closest_reference_length / self.total_hypothesis_length))

    def count_sentence_ngram(self, sentence, n):
        ngrams = dict()
        for index in range(len(sentence)-n+1):
            ngram = ' '.join(sentence[index:index+n])
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
        return ngrams

    def add(self, hypothesis, references):
        hyp_length = len(hypothesis)
        closest_ref_length, closest_length_diff = float("inf"), float("inf")
        for reference in references:
            ref_length = len(reference)
            length_diff = abs(ref_length - hyp_length)
            if length_diff < closest_length_diff:
                closest_ref_length = ref_length
                closest_length_diff = length_diff
            elif length_diff == closest_length_diff:
                closest_ref_length = min(closest_ref_length, ref_length)

        self.total_hypothesis_length += hyp_length
        self.total_closest_reference_length += closest_ref_length

        for index in range(self.gram_number):
            n = index + 1
            hyp_ngrams = self.count_sentence_ngram(hypothesis, n)
            ref_ngrams = dict()
            for reference in references:
                candidate_ref_ngrams = self.count_sentence_ngram(reference, n)
                ref_ngrams = merge_dict(ref_ngrams, candidate_ref_ngrams, restrict='max_uni')

            matched_ngrams = merge_dict(hyp_ngrams, ref_ngrams, restrict='min_its')
            self.ngram_statistics[index].correct_ngram += sum(matched_ngrams.values())
            self.ngram_statistics[index].total_ngram += sum(hyp_ngrams.values())

    @property
    def score(self):
        log_precisions = list()
        for precision in self.precisions:
            if precision == 0:
                log_precision = float("-inf")
            else:
                log_precision = math.log(precision)
            log_precisions.append(log_precision)
        weighted_precision = sum(log_precisions) / self.gram_number
        return self.brevity_penalty * math.exp(weighted_precision)

    @property
    def precisions(self):
        precisions = list()
        for index in range(self.gram_number):
            precisions.append(self.ngram_statistics[index].correct_ngram / self.ngram_statistics[index].total_ngram)
        return precisions

    @property
    def length_ratio(self):
        return self.total_hypothesis_length / self.total_closest_reference_length

    @property
    def hypothesis_length(self):
        return self.total_hypothesis_length

    @property
    def reference_length(self):
        return self.total_closest_reference_length

    @property
    def result_string(self):
        precision_str = str()
        for n, precision in enumerate(self.precisions):
            precision_str += f'P{n+1}={precision * 100:.1f} '

        res_str = (
            f"BLEU = {self.score * 100:.2f}, {precision_str}| "
            f"BP={self.brevity_penalty:.3f}, Len_Ratio={self.length_ratio:.3f}, "
            f"Hyp_Len={self.hypothesis_length}, Ref_Len={self.reference_length}"
        )
        return res_str
