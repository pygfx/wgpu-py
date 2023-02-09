# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import re
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

import wgpu  # noqa: E402
import wgpu.gui  # noqa: E402


# -- Tests -------------------------------------------------------------------

# Ensure that all classes are references in the alphabetic list,
# and referenced at least one other time as part of the explanatory text.
with open(os.path.join(ROOT_DIR, "docs", "wgpu.rst"), "rb") as f:
    wgpu_text = f.read().decode()
    wgpu_lines = [line.strip() for line in wgpu_text.splitlines()]
for cls_name in wgpu.base.__all__:
    assert f"~{cls_name}" in wgpu_lines, f"Class {cls_name} not listed in class list in wgpu.rst"
    assert f":class:`{cls_name}`" in wgpu_text, f"Class {cls_name} not referenced in the text in wgpu.rst"


# -- Hacks to tweak docstrings -----------------------------------------------

# Make flags and enums appear better in docs
wgpu.enums._use_sphinx_repr = True
wgpu.flags._use_sphinx_repr = True
wgpu.structs._use_sphinx_repr = True


func_ref_pattern = re.compile("\ (`\w+\(\)`)", re.MULTILINE)

def resolve_crossrefs(text):
    # Turn mentions of classes and methods into a cross ref (link)
    # The :obj: ref seems to work for any kind of object
    text = (text or "").lstrip()
    for start in ("GPU", "Wgpu", "gui.Wgpu"):
        text = text.replace(f" `{start}", f" :obj:`{start}")
    # Also make mentions of local methods a crossref
    while m := func_ref_pattern.search(text):
        i = m.start(1)
        text = text[:i] + ":func:" + text[i:]
    return text


# Also tweak docstrings of classes and their methods
for cls_name, cls in wgpu.base.__dict__.items():
    if cls_name not in wgpu.base.__all__:
        continue
    # Change class docstring to hide signature
    docs = resolve_crossrefs(cls.__doc__)
    cls.__doc__ = cls.__name__ + "()\n\n    " + docs
    # Change docstring of methods that dont have positional arguments
    for method in cls.__dict__.values():
        if not (callable(method) and hasattr(method, "__code__")):
            continue
        docs = resolve_crossrefs(method.__doc__)
        if method.__code__.co_argcount == 1 and method.__code__.co_kwonlyargcount > 0:
            sig = method.__name__ + "(**parameters)"
            docs = sig + "\n\n        " + docs
        method.__doc__ = docs


# -- Project information -----------------------------------------------------

project = "wgpu-py"
copyright = "2020-2023, Almar Klein, Korijn van Golen"
author = "Almar Klein, Korijn van Golen"
release = wgpu.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
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
