Installation
============

.. note:: Since the API changes with each release,you may want to check the `CHANGELOG.md <https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md>`_ when you upgrade to a newer version of wgpu.

Install with pip
----------------

You can install ``wgpu-py`` via pip.
Python 3.8 or higher is required. Pypy is supported. Only depends on ``cffi`` (installed automatically by pip).

.. code-block:: bash

    pip install wgpu


Since most users will want to render something to screen, we recommend installing GLFW as well:

.. code-block:: bash

    pip install wgpu glfw


GUI libraries
-------------

Multiple GUI backends are supported, see :doc:`the GUI API <gui>` for details:

* `glfw <https://github.com/FlorianRhiem/pyGLFW>`_: a lightweight GUI for the desktop
* `jupyter_rfb <https://jupyter-rfb.readthedocs.io>`_: only needed if you plan on using wgpu in Jupyter
* qt (PySide6, PyQt6, PySide2, PyQt5)
* wx


The wgpu-native library
-----------------------

The wheels that pip installs include the prebuilt binaries of `wgpu-native <https://github.com/gfx-rs/wgpu-native>`_, so on most systems everything Just Works.

On Linux you need at least **pip >= 20.3**, and a recent Linux distribution, otherwise the binaries will not be available. See below for details.

If you need/want, you can also `build wgpu-native yourself <https://github.com/gfx-rs/wgpu-native/wiki/Getting-Started>`_.
You will then need to set the environment variable ``WGPU_LIB_PATH`` to let wgpu-py know where the DLL is located.


Platform requirements
---------------------

Under the hood, wgpu runs on Vulkan, Metal, or DX12. The wgpu-backend
is selected automatically, but can be overridden by setting the
``WGPU_BACKEND_TYPE`` environment variable to "Vulkan", "Metal", "D3D12",
"D3D11", or "OpenGL".

Windows
+++++++

On Windows 10+, things should just work. If your machine has a dedicated GPU,
you may want to update to the latest (Nvidia or AMD) drivers.

MacOS
+++++

On MacOS you need at least 10.13 (High Sierra) to have Metal/Vulkan support.

Linux
+++++

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

Installing LavaPipe on Linux
++++++++++++++++++++++++++++

To run wgpu on systems that do not have a GPU (e.g. CI) you need a software renderer.
On Windows this (probably) just works via DX12. On Linux you can use LavaPipe:

.. code-block:: bash

        sudo apt update -y -qq
        sudo apt install --no-install-recommends -y libegl1-mesa libgl1-mesa-dri libxcb-xfixes0-dev mesa-vulkan-drivers

The distribution's version of Lavapipe may be a bit outdated. To get a more recent version, you can use this PPA:

.. code-block:: bash

        sudo add-apt-repository ppa:oibaf/graphics-drivers -y

.. note::

    The precise visual output may differ between differen implementations of Vulkan/Metal/DX12.
    Therefore you should probably avoid per-pixel comparisons when multiple different systems are
    involved. In wgpu-py and pygfx we have solved this by generating all reference images on CI (with Lavapipe).
