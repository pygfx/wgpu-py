[![Build Status](https://dev.azure.com/almarklein/wgpu-py/_apis/build/status/almarklein.wgpu-py?branchName=master)](https://dev.azure.com/almarklein/wgpu-py/_build/latest?definitionId=1&branchName=master)

# wgpu-py

Next generation GPU API for Python


## Introduction

In short, this is a Python lib wrapping the Rust wgpu lib and exposing
it with a Pythonic API similar to WebGPU.

The OpenGL API is old and showing it's cracks. New API's like Vulkan,
Metal and DX12 provide a modern way to control the GPU, but these API's
are too low-level for general use. The WebGPU API follows the same concepts, but with
a simpler (higher level) spelling. The Python `wgpu` library brings the
WebGPU API to Python. Based on [wgpu-native](https://github.com/gfx-rs/wgpu).

To get an idea of what this API looks like have a look at [triangle.py](https://github.com/almarklein/wgpu-py/blob/master/examples/triangle.py) and the other [examples](https://github.com/almarklein/wgpu-py/blob/master/examples/).


## Status

*This is experimental, work in progress, you probably don't want to use this just yet!*

* We have a few working examples!
* Support for Windows and Linux. Support for MacOS is underway.
* We have not fully implemented the API yet.
* The API may change. We're still figuring out what works best.
* The API may change more. Until WebGPU settles as a standard, its specification
  may change, and with that our API will probably too.


## Installation

```
pip install wgpu
pip install python-shader  # optional - our examples use this to define shaders
```

The library ships with Rust binaries for Windows, MacOS and Linux. If you want to use
a custom build instead, you can set the environment variable `WGPU_LIB_PATH`.


## Platform requirements

Under the hood, `wgpu` runs on Vulkan or Metal, and eventually also DX12 or OpenGL.

On Windows 10, things should just work. On older Windows versions you may need
to install the Vulkan drivers (or wait for the DX12 backend to become more mature).

On Linux, it's advisable to install the proprietary drivers of your GPU
(if you have a dedicated GPU). You may need to `apt install mesa-vulkan-drivers`.
Note that on Linux, the `tk` canvas does not work. Wayland currently only
works with the GLFW canvas (and is unstable).

On MacOS you need at least 10.13 (High Sierra) to have Vulkan support.
At the moment, we've not implemented drawing to a window yet (see #29).


## Usage

The full API is accessable via the main namespace:
```py
import wgpu
```

But to use it, you need to select a backend first. You do this by importing it.
There is currently only one backend:
```py
import wgpu.backend.rs
```


## GUI integration

To render to the screen you can use a variety of GUI toolkits:

```py
# Tk is included with Python (though this canvas only seems to work on Windows)
from wgpu.gui.tk import WgpuCanvas

# GLFW is a great lightweight windowing toolkit. Install with `pip install glfw`
from wgpu.gui.glfw import WgpuCanvas

# Visualizations can be embedded as a widget in a Qt application.
# Import PySide2, PyQt5, PySide or PyQt4 before running the line below.
# The code will detect and use the library that is imported.
from wgpu.gui.qt import WgpuCanvas
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
