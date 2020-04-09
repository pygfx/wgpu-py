---------------
Getting started
---------------

Installation
------------

``pip install wgpu``


Dependencies
------------

Python 3.6 or higher is required. Pypy is supported.

We depend on modern low-level graphics API's (Vulkan, Metal, DX12). This should
be fine on a relatively modern OS and hardware:

* Windows: Vulkan must be available (fine on Windows 10), probably
  older Windows versions too when DX12 can be used.
* MacOS: Metal must be available (10.13 High Sierra or higher).
* Linux: Vulkan must be available.

The wgpu-py package depends on `wgpu-native <https://github.com/gfx-rs/wgpu>`_,
a library written in Rust. This library is included with ``wgpu-py``.

Other than that, the only dependency is ``cffi``.
