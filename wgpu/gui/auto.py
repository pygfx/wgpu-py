"""
Automatic GUI backend selection.

Right now we only chose between GLFW and Jupyter. We might add suport
for e.g. Qt later. Or we might decide to stick with these two.
"""

__all__ = ["WgpuCanvas", "run", "call_later"]

import sys


def is_jupyter():
    """Determine whether the user is executing in a Jupyter Notebook / Lab."""
    try:
        ip = get_ipython()
        if ip.has_trait("kernel"):
            return True
        else:
            return False
    except NameError:
        return False


if is_jupyter():
    from .jupyter import WgpuCanvas, run, call_later  # noqa
else:
    try:
        from .glfw import WgpuCanvas, run, call_later  # noqa
    except ImportError as err:
        msg = str(err)
        msg += "\n\n  Install glfw using e.g. ``pip install -U glfw``."
        if sys.platform.startswith("linux"):
            msg += "\n  You may also need to run the equivalent of ``apt install libglfw3``."
        raise ImportError(msg) from None
