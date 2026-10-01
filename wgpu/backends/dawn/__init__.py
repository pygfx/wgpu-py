"""
The Dawn backend (experimental).

Dawn is the WebGPU implementation of Chromium. This backend wraps Dawn's
``webgpu.h`` C API using a compiled Cython extension module. It is optional:
it must be built against an installed Dawn (e.g. ``conda install -c conda-forge dawn``)
with ``python tools/build_dawn.py``.

It can also be built for Pyodide (see ``tools/build_dawn.py``), where it is
the default backend if installed.

To use it, import it before anything else selects a backend::

    import wgpu.backends.dawn

or set the ``WGPUPY_BACKEND=dawn`` environment variable.
"""

# ruff: noqa: F401, F403

try:
    from . import _api
except ImportError as err:  # no-cover
    raise ImportError(
        "The Dawn backend is not available: its extension module is not built "
        "or Dawn (libwebgpu_dawn) cannot be loaded. Build it with "
        "'python tools/build_dawn.py' (requires Dawn, Cython and a C compiler). "
        f"Original error: {err}"
    ) from err

from ._api import *
from ._api import process_events, get_instance_address
from .. import _register_backend


# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)
