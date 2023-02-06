---------------
Getting started
---------------

Installation
------------

.. code-block:: bash

    pip install wgpu


Dependencies
------------

Python dependencies:

* Python 3.7 or higher is required. Pypy is supported.
* Only depends on ``cffi`` (installed automatically by pip).

The wgpu-native library:

* The wheels include the prebuilt binaries of `wgpu-native <https://github.com/gfx-rs/wgpu-native>`_.
* On Linux you need at least **pip >= 20.3**, and a recent Linux distribution, otherwise the binaries will not be available. See *platform requirements* for details.
* If you need/want to `build wgpu-native yourself <https://github.com/gfx-rs/wgpu-native/wiki/Getting-Started>`_, you need to set the environment variable ``WGPU_LIB_PATH`` to let wgpu-py know where the DLL is located.


GUI libraries
-------------

Most users will want to render something to screen. To this end we recommend:

.. code-block:: bash

    pip install glfw

wgpu-py supports:

* `glfw <https://github.com/FlorianRhiem/pyGLFW>`_: a lightweight GUI for the desktop
* `jupyter_rfb <https://jupyter-rfb.readthedocs.io/en/latest/>`_: only needed if you plan on using wgpu in Jupyter
* qt (PySide6, PyQt6, PySide2, PyQt5)
* wx


Platform requirements
---------------------

Under the hood, wgpu runs on Vulkan, Metal, or DX12. The wgpu-backend
is selected automatically, but can be overridden by setting the
``WGPU_BACKEND_TYPE`` environment variable to "Vulkan", "Metal", "D3D12",
"D3D11", or "OpenGL".

Windows
=======

On Windows 10+, things should just work. On older Windows versions you
may need to install the Vulkan drivers.

MacOS
=====

On MacOS you need at least 10.13 (High Sierra) to have Metal/Vulkan support.

Linux
=====

On Linux, it's advisable to install the proprietary drivers of your GPU
(if you have a dedicated GPU). You may need to ``apt install
mesa-vulkan-drivers``. Wayland support is currently broken (we could use
a hand to fix this).

Binary wheels for Linux are only available for **manylinux_2_24**.
This means that the installation requires ``pip >= 20.3``, and you need
a recent Linux distribution, listed `here <https://github.com/pypa/manylinux#manylinux>`_.

If you wish to work with an older distribution, you will have to build
wgpu-native yourself, see "dependencies" above. Note that wgpu-native
still needs Vulkan support and may not compile / work on older
distributions.


About this API
--------------

This library presents a Pythonic API for the `WebGPU spec
<https://gpuweb.github.io/gpuweb/>`_. It is an API to control graphics
hardware. Like OpenGL but modern, or like Vulkan but higher level.
GPU programming is a craft that requires knowledge of how GPU's work.
See the guide for more info and links to resources.


What's new in this version?
---------------------------

Since the API changes with each release, and we do not yet make things
backwards compatible. You may want to check the `CHANGELOG.md <https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md>`_
when you upgrade to a newer version of wgpu.
