[![CI](https://github.com/pygfx/wgpu-py/workflows/CI/badge.svg)](https://github.com/pygfx/wgpu-py/actions)
[![Documentation Status](https://readthedocs.org/projects/wgpu-py/badge/?version=stable)](https://wgpu-py.readthedocs.io)
[![PyPI version](https://badge.fury.io/py/wgpu.svg)](https://badge.fury.io/py/wgpu)
[![Zenodo badge](https://zenodo.org/badge/DOI/10.5281/zenodo.18836262.svg)](https://doi.org/10.5281/zenodo.18836262)


# wgpu-py

A Python implementation of WebGPU - the next generation GPU API. 🚀

<div>
  <img width=320 src='https://raw.githubusercontent.com/pygfx/wgpu-py/main/examples/screenshots/triangle.png' />
  <img width=320 src='https://raw.githubusercontent.com/pygfx/wgpu-py/main/examples/screenshots/cube.png' />
</div>


## Introduction

The purpose of wgpu-py is to provide Python with a powerful and reliable GPU API.

It serves as a basis to build a broad range of applications and libraries
related to visualization and GPU compute. We use it in
[pygfx](https://github.com/pygfx/pygfx) to create a modern Pythonic render
engine.

To get an idea of what this API looks like have a look at
[triangle.py](https://github.com/pygfx/wgpu-py/blob/main/examples/triangle.py)
and the other [examples](https://github.com/pygfx/wgpu-py/blob/main/examples/).


## Status

* Until WebGPU settles as a standard, its specification may change, and with
  that our API will probably too. Check the [changelog](CHANGELOG.md) when you
  upgrade!
* Coverage of the WebGPU spec is complete enough to build e.g.
  [pygfx](https://github.com/pygfx/pygfx).
* Test coverage of the API is close to 100%.
* Support for Windows, Linux (x86 and aarch64), and MacOS (Intel and M1).


## What is WebGPU / wgpu?

WGPU is the future for GPU graphics; the successor to OpenGL.

WebGPU is a JavaScript API with a well-defined
[spec](https://gpuweb.github.io/gpuweb/), the successor to WebGL. The somewhat
broader term "wgpu" is used to refer to "desktop" implementations of WebGPU in
various languages.

OpenGL is old and showing its cracks. New API's like Vulkan, Metal and DX12
provide a modern way to control the GPU, but these are too low-level for general
use. WebGPU follows the same concepts, but with a simpler (higher level) API.
With wgpu-py we bring WebGPU to Python.

Technically speaking, wgpu-py is a wrapper for
[wgpu-native](https://github.com/gfx-rs/wgpu-native), exposing its functionality with a Pythonic
API closely resembling the [WebGPU spec](https://gpuweb.github.io/gpuweb/).


## Installation

```
# Just wgpu
pip install wgpu

# If you want to render to screen
pip install wgpu rendercanvas glfw
```

Linux users should make sure that **pip >= 20.3**. That should do the
trick on most systems. See [getting started](https://wgpu-py.readthedocs.io/en/stable/start.html)
for details.


## Usage

Also see the [online documentation](https://wgpu-py.readthedocs.io) and the [examples](https://github.com/pygfx/wgpu-py/tree/main/examples).

The full API is accessible via the main namespace:
```py
import wgpu
```

To render to the screen you can use a variety of GUI toolkits:

```py
# The rendercanvas auto backend selects either the glfw, qt, wx, or jupyter backend
from rendercanvas.auto import RenderCanvas, loop

# Visualizations can be embedded as a widget in a Qt application.
# Import PySide6, PyQt6, PySide2 or PyQt5 before running the line below.
# The code will detect and use the library that is imported.
from rendercanvas.qt import RenderCanvas
```

Some functions in the original `wgpu-native` API are async. In the Python API,
the default functions are all sync (blocking), making things easy for general use.
Async versions of these functions are available, so wgpu can also work
well with Asyncio or Trio.


## License

This code is distributed under the 2-clause BSD license.


## Projects using `wgpu-py`

* [pygfx](https://github.com/pygfx/pygfx) - A python render engine running on wgpu.
* [shadertoy](https://github.com/pygfx/shadertoy) - Shadertoy implementation using wgpu-py.
* [tinygrad](https://github.com/tinygrad/tinygrad) - deep learning framework
* [fastplotlib](https://github.com/fastplotlib/fastplotlib) - A fast plotting library
* [xdsl](https://github.com/xdslproject/xdsl) - A Python Compiler Design Toolkit (optional wgpu interpreter)


## Contributing

See the [contribution guide](CONTRIBUTING.md).

### Development install

* Clone the repo.
* Install devtools using `pip install -e .[dev]`.
* Using `pip install -e .` will also download the upstream wgpu-native binaries.
  * You can use `python tools/download_wgpu_native.py` when needed. And run `python codegen` once to obtain the combined headerfile.
  * Or point the `WGPU_LIB_PATH` environment variable to a custom build of `wgpu-native`.

### Quick tips

* Use `ruff format` to apply autoformatting.
* Use `ruff check` to check for linting errors.
* Use `pytest -v tests` runs the unit tests.
* Use `pytest -v examples` tests the examples.

### Code of Conduct

This repository follows the [PyGfx Code of Conduct](https://github.com/pygfx/pygfx/blob/main/CODE_OF_CONDUCT.md)

