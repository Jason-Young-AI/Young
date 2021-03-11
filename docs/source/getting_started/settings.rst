.. _Command-line Tools:

Command-line Tools
==================

Fairseq provides several command-line tools for training and evaluating models:

- :ref:`fairseq-preprocess`: Data pre-processing: build vocabularies and binarize training data
- :ref:`fairseq-train`: Train a new model on one or multiple GPUs
- :ref:`fairseq-generate`: Translate pre-processed data with a trained model
- :ref:`fairseq-interactive`: Translate raw text with a trained model
- :ref:`fairseq-score`: BLEU scoring of generated translations against reference translations
- :ref:`fairseq-eval-lm`: Language model evaluation


.. _fairseq-preprocess:

fairseq-preprocess
~~~~~~~~~~~~~~~~~~
.. automodule:: fairseq_cli.preprocess

    .. argparse::
        :module: fairseq.options
        :func: get_preprocessing_parser
        :prog: fairseq-preprocess


.. _fairseq-train:

fairseq-train
~~~~~~~~~~~~~
.. automodule:: fairseq_cli.train

    .. argparse::
        :module: fairseq.options
        :func: get_training_parser
        :prog: fairseq-train


.. _fairseq-generate:

fairseq-generate
~~~~~~~~~~~~~~~~
.. automodule:: fairseq_cli.generate

    .. argparse::
        :module: fairseq.options
        :func: get_generation_parser
        :prog: fairseq-generate

