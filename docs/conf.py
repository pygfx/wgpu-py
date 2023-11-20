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
import shutil


ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

import wgpu  # noqa: E402


# -- Tests -------------------------------------------------------------------

# Ensure that all classes are referenced in the alphabetic list,
# and referenced at least one other time as part of the explanatory text.
with open(os.path.join(ROOT_DIR, "docs", "wgpu.rst"), "rb") as f:
    wgpu_text = f.read().decode()
    wgpu_lines = [line.strip() for line in wgpu_text.splitlines()]
for cls_name in wgpu.classes.__all__:
    assert (
        f"~{cls_name}" in wgpu_lines
    ), f"Class '{cls_name}' not listed in class list in wgpu.rst"
    assert (
        f":class:`{cls_name}`" in wgpu_text
    ), f"Class '{cls_name}' not referenced in the text in wgpu.rst"


# -- Hacks to tweak docstrings -----------------------------------------------

# Make flags and enums appear better in docs
wgpu.enums._use_sphinx_repr = True
wgpu.flags._use_sphinx_repr = True
wgpu.structs._use_sphinx_repr = True

# Build regular expressions to resolve crossrefs
func_ref_pattern = re.compile(r"\ (`\w+?\(\)`)", re.MULTILINE)
ob_ref_pattern = re.compile(
    r"\ (`(GPU|gui\.Wgpu|flags\.|enums\.|structs\.)\w+?`)", re.MULTILINE
)
argtype_ref_pattern = re.compile(
    r"\(((GPU|gui\.Wgpu|flags\.|enums\.|structs\.)\w+?)\)", re.MULTILINE
)


def resolve_crossrefs(text):
    text = (text or "").lstrip()

    # Turn references to functions into a crossref.
    # E.g. `Foo.bar()`
    i2 = 0
    while True:
        m = func_ref_pattern.search(text, i2)
        if not m:
            break
        i1, i2 = m.start(1), m.end(1)
        ref_indicator = ":func:"
        text = text[:i1] + ref_indicator + text[i1:]

    # Turn references to objects (classes, flags, enums, and structs) into a crossref.
    # E.g. `GPUDevice` or `flags.BufferUsage`
    i2 = 0
    while True:
        m = ob_ref_pattern.search(text, i2)
        if not m:
            break
        i1, i2 = m.start(1), m.end(1)
        prefix = m.group(2)  # e.g. GPU or flags.
        ref_indicator = ":obj:" if prefix.lower() == prefix else ":class:"
        text = text[:i1] + ref_indicator + text[i1:]

    # Turn function arg types into a crossref.
    # E.g. (GPUDevice) or (flags.BufferUsage)
    i2 = 0
    while True:
        m = argtype_ref_pattern.search(text)
        if not m:
            break
        i1, i2 = m.start(1), m.end(1)
        ref_indicator = ":obj:"
        text = text[:i1] + ref_indicator + "`" + text[i1:i2] + "`" + text[i2:]

    return text


# Tweak docstrings of classes and their methods
for module, hide_class_signature in [(wgpu.classes, True), (wgpu.gui, False)]:
    for cls_name in module.__all__:
        cls = getattr(module, cls_name)
        # Class docstring
        docs = resolve_crossrefs(cls.__doc__)
        if hide_class_signature:
            docs = cls.__name__ + "()\n\n    " + docs
        cls.__doc__ = docs or None
        # Docstring of methods
        for method in cls.__dict__.values():
            if callable(method) and hasattr(method, "__code__"):
                docs = resolve_crossrefs(method.__doc__)
                if (
                    method.__code__.co_argcount == 1
                    and method.__code__.co_kwonlyargcount > 0
                ):
                    sig = method.__name__ + "(**parameters)"
                    docs = sig + "\n\n        " + docs
                method.__doc__ = docs or None


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

# Just let autosummary produce a new version each time
shutil.rmtree(os.path.join(os.path.dirname(__file__), "generated"), True)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

if not (os.getenv("READTHEDOCS") or os.getenv("CI")):
    html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
