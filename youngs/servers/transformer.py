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


import os
import re
import json
import torch
import flask
import wtforms
import requests
import flask_wtf

from subword_nmt.apply_bpe import BPE

from youngs.servers import register_server, Server

from youngs.data.batch import Batch
from youngs.data.instance import Instance

from youngs.utilities.sequence import stringize, tokenize, numericalize, dehyphenate


@register_server('transformer')
class Transformer(Server):
    def __init__(self,
        apply_bpe, bpe_codes,
        web_secret_key, app_secret_key,
        web_host, web_port,
        app_host, app_port,
        tester, logger
    ):
        super(Transformer, self).__init__(
            web_host, web_port,
            app_host, app_port,
            tester, logger
        )

        self.apply_bpe = apply_bpe
        self.bpe_codes = bpe_codes
        self.web_secret_key = web_secret_key
        self.app_secret_key = app_secret_key

    @classmethod
    def setup(cls, settings, tester, logger):
        args = settings.args
        server = cls(
            args.apply_bpe, args.bpe_codes,
            args.web.secret_key,
            args.app.secret_key,
            args.web.host, args.web.port,
            args.app.host, args.app.port,
            tester, logger
        )

        return server

    @property
    def web_end(self):
        import_name = '.'.join(__name__.split('.')[:-1])
        web_end = flask.Flask(import_name)
        web_end.config['SECRET_KEY'] = self.web_secret_key

        app_end_address = f'http://{self.app_host}:{self.app_port}'

        class TranslateForm(flask_wtf.FlaskForm):
            source = wtforms.TextAreaField('Type Here', validators=[wtforms.validators.DataRequired()])
            submit = wtforms.SubmitField('Translate')

        @web_end.route('/', methods=['GET', 'POST'])
        def index():
            form = TranslateForm()
            target = None

            response = requests.get(f'{app_end_address}/languages')
            json_data = json.loads(response.text)
            source_language = json_data['source_language']
            target_language = json_data['target_language']

            if form.validate_on_submit():
                source = form.source.data

                json_data = {"source": source}
                headers = {"Content-Type": "application/json"}

                response = requests.post(f'{app_end_address}/translate', json=json_data, headers=headers)
                json_data = json.loads(response.text)

                target = json_data['translation']
            else:
                target = 'Translation'

            return flask.render_template(
                'transformer.html',
                form=form,
                source_language=source_language,
                target_language=target_language,
                target=target
            )

        return web_end

    @property
    def app_end(self):
        import_name = '.'.join(__name__.split('.')[:-1])
        app_end = flask.Flask(import_name)
        app_end.config['SECRET_KEY'] = self.app_secret_key
        if self.apply_bpe:
            bpe = BPE(open(self.bpe_codes, encoding='utf-8'), separator=self.tester.bpe_symbol)

        def preprocess(source):
            source_sentences = source.strip().split('\n')
            instances = list()
            for source_sentence in source_sentences:
                if self.apply_bpe:
                    source_sentence = bpe.process_line(source_sentence)
                source_attribute = numericalize(tokenize(source_sentence), self.tester.factory.vocabularies['source'], add_bos=True, add_eos=True)
                instances.append(Instance(source=source_attribute))

            batch = Batch(set({'source', }))
            for instance in instances:
                batch.add(instance)
            return batch

        def postprocess(result):
            candidate_paths = result
            parallel_line_number = len(candidate_paths)

            candidate_translations = list()
            top1_translations = list()
            for line_index in range(parallel_line_number):

                candidate_path_number = len(candidate_paths[line_index])
                for path_index in range(candidate_path_number):
                    candidate_path = candidate_paths[line_index][path_index]

                    lprob = candidate_path['log_prob']
                    score = candidate_path['score']
                    trans = candidate_path['path']

                    trans_tokens = stringize(trans, self.tester.factory.vocabularies['target'])
                    trans_sentence = ' '.join(trans_tokens)

                    if self.tester.remove_bpe:
                        trans_sentence = (trans_sentence + ' ').replace(f"{self.tester.bpe_symbol} ", '').strip()
                    if self.tester.dehyphenate:
                        trans_sentence = dehyphenate(trans_sentence)
                    candidate_translation = dict(
                        lprob = lprob,
                        score = score,
                        trans = trans_sentence,
                    )
                    candidate_translations.append(candidate_translation)
                    if path_index == 0:
                        top1_translations.append(trans_sentence)

            processed_result = dict(
                translation = '\n'.join(top1_translations),
                detailed_translation = candidate_translations
            )
            return processed_result

        @app_end.route(f'/languages', methods=['GET'])
        def get_languages():
            json_data = flask.jsonify({'source_language': self.tester.factory.source_language, 'target_language': self.tester.factory.target_language})
            return json_data

        @app_end.route(f'/translate', methods=['POST'])
        def translate():
            json_data = flask.request.get_json(force=True)
            source = json_data['source']
            batch = preprocess(source)

            customized_batch = self.tester.customize_batch(batch)
            result = self.tester.test_batch(customized_batch)

            processed_result = postprocess(result)
            json_data = flask.jsonify(processed_result)
            return json_data

        return app_end
