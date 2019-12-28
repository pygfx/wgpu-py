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
* This currently only works on Windows.
* We don't yet package the wgpu lib; you have to bring it along yourself for now.
* We have not fully implemented the API yet.
* The API may change. We're still figuring out what works best.
* The API may change more. Until WebGPU settles as a standard, its specification
  may change, and with that our API will probably too.


## Installation

```
pip install wgpu
pip install spirv  # optional - our examples use this to define shaders
```

This library will eventually include the required Rust library, but for
now, you have to bring it yourself. Tell where it is by setting the
environment variable `WGPU_LIB_PATH`.


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

To render to the screen you can use any of the following GUI toolkits:
`Tk` (included with Python), `glfw`, `PySide2`, `PyQt5`.


## Web support

We are considering future support for compiling (Python)
visualizations to the web via PScript and Flexx. We try to keep that
option open as long as it does not get in the way too much. No promises.


## License

This code is distributed under the 2-clause BSD license.


## Developers

* Clone the repo and run `python setup.py develop`, or simply add the root dir to your `PYTHONPATH`.
* Point the `WGPU_LIB_PATH` environment variable to the dynamic library created by `wgpu-native`.
* Use `black .` to apply autoformatting.
* Use `flake8 .` to check for flake errors.
* Use `pytest .` to run the tests.
