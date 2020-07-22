#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-03-31 22:05
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from ynmt.models.transformer import build_model_transformer


def build_model(args, vocabularies, checkpoint, device_descriptor):
    model = globals()[f'build_model_{args.name}'](args, vocabularies)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device_descriptor)
    return model
