# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

sys.path.insert(0, os.path.abspath('../../'))

with open(os.path.join("youngs", "version")) as version_file:
    version = version_file.read().strip()

project = 'YoungNMT'
copyright = 'Jason Young (杨郑鑫)'
author = 'Jason Young (杨郑鑫)'
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',  # Include documentation from docstrings
        'sphinx.ext.napoleon', # Support for NumPy and Google style docstrings
        'sphinx.ext.viewcode', # Add links to highlighted source code
        'sphinx.ext.coverage', # Collect doc coverage stats
        'sphinx.ext.githubpages', # Publish HTML docs in GitHub Pages
        'sphinx_rtd_theme',
]

source_suffix = {
        '.rst': 'restructuredtext',
}

master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/youngs_logo.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

github_doc_root = 'https://github.com/Jason-Young-AI/YoungNMT/tree/master/doc/
