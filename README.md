![CI](https://github.com/pygfx/wgpu-py/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/wgpu-py/badge/?version=latest)](https://wgpu-py.readthedocs.io)

# wgpu-py

Next generation GPU API for Python


## Introduction

In short, this is a Python lib wrapping [wgpu-native](https://github.com/gfx-rs/wgpu)
and exposing it with a Pythonic API similar to the [WebGPU spec](https://gpuweb.github.io/gpuweb/).

The OpenGL API is old and showing it's cracks. New API's like Vulkan,
Metal and DX12 provide a modern way to control the GPU, but these API's
are too low-level for general use. The WebGPU API follows the same concepts, but with
a simpler (higher level) spelling. The Python `wgpu` library brings the
WebGPU API to Python.

To get an idea of what this API looks like have a look at [triangle.py](https://github.com/pygfx/wgpu-py/blob/main/examples/triangle.py) and the other [examples](https://github.com/pygfx/wgpu-py/blob/main/examples/).


## Status

*The wgpu-API has not settled yet, use with care!*

* Coverage of the WebGPU spec is nearly complete.
* Test coverage of the API is 100%.
* Support for Windows, Linux and MacOS.
* Until WebGPU settles as a standard, its specification
  may change, and with that our API will probably too. Check the [changelog](CHANGELOG.md) when you upgrade!


## Installation

```
pip install wgpu
```

The library ships with Rust binaries for Windows, MacOS and Linux. If you want to use
a custom build instead, you can set the environment variable `WGPU_LIB_PATH`.


## Platform requirements

Under the hood, `wgpu` runs on Vulkan, Metal, or DX12. The wgpu-backend is selected automatically, but can be overridden by setting the `WGPU_BACKEND_TYPE` environment variable to "Vulkan", "Metal", "D3D12", "D3D11", or "OpenGL".

On Windows 10, things should just work. On older Windows versions you
may need to install the Vulkan drivers. You may want to force "Vulkan" while "D3D12" is less mature.

On Linux, it's advisable to install the proprietary drivers of your GPU
(if you have a dedicated GPU). You may need to `apt install mesa-vulkan-drivers`.
Wayland currently only works with the GLFW canvas (and is unstable).

On MacOS you need at least 10.13 (High Sierra) to have Vulkan support.


## Usage

Also see the [online documentation](https://wgpu-py.readthedocs.io).

The full API is accessable via the main namespace:
```py
import wgpu
```

But to use it, you need to select a backend first. You do this by importing it.
There is currently only one backend:
```py
import wgpu.backend.rs
```

To render to the screen you can use a variety of GUI toolkits:

```py
# GLFW is a great lightweight windowing toolkit. Install with `pip install glfw`
from wgpu.gui.glfw import WgpuCanvas

# Visualizations can be embedded as a widget in a Qt application.
# Import PySide2, PyQt5, PySide or PyQt4 before running the line below.
# The code will detect and use the library that is imported.
from wgpu.gui.qt import WgpuCanvas

# You can also show wgpu visualizations in Jupyter
from wgpu.gui.jupyter import WgpuCanvas
```

Some functions in the original `wgpu-native` API are async. In the Python API,
the default functions are all sync (blocking), making things easy for general use.
Async versions of these functions are available, so wgpu can also work
well with Asyncio or Trio.


## Web support

We are considering future support for compiling (Python)
visualizations to the web via PScript and Flexx. We try to keep that
option open as long as it does not get in the way too much. No promises.


## License

This code is distributed under the 2-clause BSD license.


## Developers

* Clone the repo.
* Install devtools using `pip install -r dev-requirements.txt` (you can replace `pip` with `pipenv` to install to a virtualenv).
* Install wgpu-py in editable mode by running `python setup.py develop`, this will also install our only runtime dependency `cffi`
* Run `python download-wgpu-native.py` to download the upstream wgpu-native binaries.
  * Or alternatively point the `WGPU_LIB_PATH` environment variable to a custom build.
* Use `black .` to apply autoformatting.
* Use `flake8 .` to check for flake errors.
* Use `pytest .` to run the tests.


### Changing the upstream wgpu-native version

* Use the optional arguments to `python download-wgpu-native.py --help` to download a different version of the upstream wgpu-native binaries.
* The file `wgpu/resources/wgpu_native-version` will be updated by the script to track which version we depend upon.
