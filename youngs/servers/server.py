#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-12-16 02:16
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch
import waitress


class Server(object):
    def __init__(self,
        web_host, web_port,
        app_host, app_port,
        tester, logger
    ):
        self.web_host = web_host
        self.web_port = web_port
        self.app_host = app_host
        self.app_port = app_port
        self.tester = tester
        self.logger = logger

    def launch(self, serve_type):
        assert serve_type in set({'web', 'app'}), f'Wrong type of serve_type: [\'{serve_type}\']'

        if serve_type == 'app':
            self.tester.model.train(False)
            with torch.no_grad():
                waitress.serve(self.app_end, host=self.app_host, port=self.app_port)

        if serve_type == 'web':
            waitress.serve(self.web_end, host=self.web_host, port=self.web_port)

    @classmethod
    def setup(cls, settings, tester, logger):
        raise NotImplementedError

    @property
    def web_end(self):
        raise NotImplementedError

    @property
    def app_end(self):
        raise NotImplementedError
