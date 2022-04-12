---------------
Getting started
---------------

Installation
------------

.. code-block:: bash

    pip install wgpu


Dependencies
------------

* Python 3.7 or higher is required. Pypy is supported.
* The required ``wgpu-native`` library is distributed as part of the ``wgpu-py`` package.
* The only other dependency is ``cffi`` (installed automatically by pip).


System requirements
-------------------

The system must be new enough to support Metal or Vulkan:

* Windows: fine on Windows 10, probably older Windows versions too when DX12 can be used.
* MacOS: version 10.13 High Sierra or higher.
* Linux: Vulkan must be available.


About this API
--------------

This library presents a Pythonic API for the `WebGPU spec
<https://gpuweb.github.io/gpuweb/>`_. It is an API to control graphics
hardware. Like OpenGL, but modern.
GPU programming is a craft that requires knowledge of how GPU's work.
See the guide for more info and links to resources.


What's new in this version?
---------------------------

Since the API changes with each release, and we do not yet make things
backwards compatible. You may want to check the changelog when you
upgrade to a newer version of wgpu:

https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md
