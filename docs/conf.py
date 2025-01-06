# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import toml

confpath = os.path.dirname(__file__)
rootpath = os.path.realpath(os.path.join(confpath, ".."))
sys.path.append(rootpath)  # so that this script can find komplot module
os.environ["PYTHONPATH"] = rootpath  # so that nbsphinx can find komplot module

from komplot import _package_version

pprjpath = os.path.join(rootpath, "pyproject.toml")
pptml = toml.load(pprjpath)

project = pptml["project"]["name"]
author = pptml["project"]["authors"][0]["name"]
try:
    release = pptml["project"]["version"]
except KeyError:
    release = _package_version()
version = project + " " + release
copyright = "2024-2025, " + author


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

pygments_style = "sphinx"

plot_html_show_formats = False

autodoc_default_options = {
    "member-order": "bysource",
    "inherited-members": False,
    "ignore-module-all": False,
    "show-inheritance": True,
    "members": True,
    "special-members": "__call__",
}
autodoc_member_order = "bysource"
autodoc_docstring_signature = True
autodoc_typehints = "both"
autoclass_content = "both"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_use_ivar = True
napoleon_use_rtype = False

nbsphinx_prolog = """
.. raw:: html

    <style>
    .nbinput .prompt, .nboutput .prompt {
        display: none;
    }
    div.highlight {
        background-color: #f9f9f4;
    }
    p {
        margin-bottom: 0.8em;
        margin-top: 0.8em;
    }
    </style>
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "top_of_page_button": None,
    "sidebar_hide_name": True,
}
html_static_path = ["_static"]

html_logo = "_static/logo.png"

html_css_files = ["komplot.css"]
