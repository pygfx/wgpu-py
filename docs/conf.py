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
import subprocess


ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "true"


# Load wgpu so autodoc can query docstrings
import wgpu  # noqa: E402
import wgpu.utils.compute  # noqa: E402
import wgpu.utils.glfw_present_info  # noqa: E402


# -- Tests -------------------------------------------------------------------

# Ensure that all classes are referenced in the alphabetic list,
# and referenced at least one other time as part of the explanatory text.
with open(os.path.join(ROOT_DIR, "docs", "wgpu.rst"), "rb") as f:
    wgpu_text = f.read().decode()
    wgpu_lines = [line.strip() for line in wgpu_text.splitlines()]
for cls_name in wgpu.classes.__all__:
    assert f"~{cls_name}" in wgpu_lines, (
        f"Class '{cls_name}' not listed in class list in wgpu.rst"
    )
    assert f":class:`{cls_name}`" in wgpu_text, (
        f"Class '{cls_name}' not referenced in the text in wgpu.rst"
    )


# -- Hacks to tweak docstrings -----------------------------------------------

# Make structs appear better in docs
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
for module, hide_class_signature in [(wgpu.classes, True)]:
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
copyright = "2020-2025, Almar Klein, Korijn van Golen"
author = "Almar Klein, Korijn van Golen"
release = wgpu.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
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

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# copied and adapted from the rendercanvas

# -- Build wheel so Pyodide examples can use exactly this version of wgpu -----------------------------------------------------

short_version = ".".join(str(i) for i in wgpu.version_info[:3])
wheel_name = f"wgpu-{short_version}-py3-none-any.whl"

# Build the wheel
os.environ["WGPU_BUILD_PLATFORM_INFO"] = " ".join(["pyodide_wasm", "any"])
subprocess.run([sys.executable, "-m", "build", "-nw"], cwd=ROOT_DIR)
wheel_filename = os.path.join(ROOT_DIR, "dist", wheel_name)
assert os.path.isfile(wheel_filename), f"{wheel_name} does not exist"

# Copy into static
print("Copy wheel to _static dir")
shutil.copy(
    wheel_filename,
    os.path.join(ROOT_DIR, "docs", "_static", wheel_name),
)


# -- Sphinx Gallery -----------------------------------------------------

iframe_placeholder_rst = """
.. only:: html

    Interactive example
    ===================

    This uses Pyodide. If this does not work, your browser may not have sufficient support for wasm/pyodide/wgpu (check your browser dev console).
    Stdout (print statements) will also appear in the browser console.

    .. raw:: html

        <iframe src="pyodide.html#example.py"></iframe>
"""

python_files = {}


def add_pyodide_to_examples(app):
    if app.builder.name != "html":
        return

    gallery_dir = os.path.join(ROOT_DIR, "docs", "gallery")

    for fname in os.listdir(gallery_dir):
        filename = os.path.join(gallery_dir, fname)
        if not fname.endswith(".py"):
            continue
        with open(filename, "rb") as f:
            py = f.read().decode()
        if fname in ["cube.py", "triangle.py", "imgui_backend_sea.py", "compute_noop.py", "imgui_renderer_sea.py", "imgui_basic_example.py"]:
            # todo: later we detect by using a special comment in the py file
            print("Adding Pyodide example to", fname)
            fname_rst = fname.replace(".py", ".rst")
            # Update rst file
            rst = iframe_placeholder_rst.replace("example.py", fname)
            with open(os.path.join(gallery_dir, fname_rst), "ab") as f:
                f.write(rst.encode())
            python_files[fname] = py


def add_files_to_run_pyodide_examples(app, exception):
    if app.builder.name != "html":
        return

    gallery_build_dir = os.path.join(app.outdir, "gallery")

    # Write html file that can load pyodide examples
    with open(
        os.path.join(ROOT_DIR, "docs", "_static", "_pyodide_iframe.html"), "rb"
    ) as f:
        html = f.read().decode()
    html = html.replace('"wgpu"', f'"../_static/{wheel_name}"')
    with open(os.path.join(gallery_build_dir, "pyodide.html"), "wb") as f:
        f.write(html.encode())

    # Write the python files
    for fname, py in python_files.items():
        print("Writing", fname)
        with open(os.path.join(gallery_build_dir, fname), "wb") as f:
            f.write(py.encode())


# Suppress "cannot cache unpickable configuration value" for sphinx_gallery_conf
# See https://github.com/sphinx-doc/sphinx/issues/12300
suppress_warnings = ["config.cache"]

# The gallery conf. See https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "gallery_dirs": "gallery",
    "backreferences_dir": "gallery/backreferences",
    "doc_module": ("wgpu",),
    # "image_scrapers": (),
    "remove_config_comments": True,
    "examples_dirs": "../examples/",
    "ignore_pattern": r"serve_browser_examples\.py",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = ["style.css"]


def setup(app):
    app.connect("builder-inited", add_pyodide_to_examples)
    app.connect("build-finished", add_files_to_run_pyodide_examples)
