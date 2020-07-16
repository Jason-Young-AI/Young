#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2020-07-11 00:51
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import visdom


class Visualizer(object):
    def __init__(self, name, server, port, username=None, password=None, logging_path=None, overwrite=False):
        self.name = name
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.overwrite = overwrite
        self.logging_path = logging_path
        self.environment = None
        self.windows = set()
        self.window_layout_options = dict(
            height = 300,
            width = 400,
            showlegend = True,
            marginleft = 36,
            marginright = 36,
            margintop = 36,
            marginbottom = 36
        )

    def open(self):
        self.environment = visdom.Visdom(
            env=self.name,
            server=self.server, port=self.port,
            username=self.username, password=self.password,
            log_to_filename=self.logging_path)

        self.environment.check_connection()

        if self.logging_path is None:
            return
        else:
            if os.path.isfile(self.logging_path):
                if self.overwrite:
                    with open(self.logging_path, 'w', encoding='utf-8') as logging_file:
                        logging_file.truncate()
            else:
                os.mknod(self.logging_path)

            with open(self.logging_path) as logging_file:
                for json_entry in logging_file:
                    endpoint, msg = json.loads(json_entry)
                    window_name = self.environment._send(msg, endpoint, from_log=True)
                    self.windows.add(window_name)

    def close(self):
        if self.environment is None:
            return
        else:
            self.environment.delete_env(self.name)

    def visualize(self, visualize_type, visualize_name, visualize_title, **keyword_args):
        visualize_method = getattr(self.environment, visualize_type)

        visualize_options = dict()
        visualize_options.update(self.window_layout_options)
        visualize_options.update(keyword_args.get('opts', dict()))
        visualize_options['title'] = visualize_title
        keyword_args['opts'] = visualize_options

        visualize_method(env=self.name, win=visualize_name, **keyword_args)

        self.windows.add(visualize_name)
        return visualize_name
