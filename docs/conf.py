# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

import wgpu  # noqa: E402
import wgpu.gui  # noqa: E402


# -- Tweak wgpu's docs -------------------------------------------------------

# Ensure that our API docs are complete
with open(os.path.join(ROOT_DIR, "docs", "reference_wgpu.rst"), "rb") as f:
    wgpu_api_docs_text = f.read().decode()
for cls_name in wgpu.base.__all__:
    expected_line = f".. autoclass:: wgpu.{cls_name}\n"
    assert expected_line in wgpu_api_docs_text, f"Missing docs for {cls_name}"

# Make flags and enum appear better in docs
wgpu.enums._use_sphinx_repr = True
wgpu.flags._use_sphinx_repr = True

# Also tweak docstrings of classes and their methods
for cls_name, cls in wgpu.base.__dict__.items():
    if cls_name not in wgpu.base.__all__:
        continue

    # Change class docstring to include a link to the base class,
    # and the class' signature is not shown
    base_info = ""
    base_cls = cls.mro()[1]
    if base_cls is not object:
        base_info = f"    *Subclass of* :class:`.{base_cls.__name__}`\n\n"
    cls.__doc__ = cls.__name__ + "()\n\n" + base_info + "    " + cls.__doc__.lstrip()
    # Change docstring of methods that dont have positional arguments
    for method in cls.__dict__.values():
        if not (callable(method) and hasattr(method, "__code__")):
            continue
        if method.__code__.co_argcount == 1 and method.__code__.co_kwonlyargcount > 0:
            sig = method.__name__ + "(**parameters)"
            method.__doc__ = sig + "\n\n        " + method.__doc__.lstrip()


# -- Project information -----------------------------------------------------

project = "wgpu-py"
copyright = "2020-2021, Almar Klein, Korijn van Golen"
author = "Almar Klein, Korijn van Golen"
release = wgpu.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
