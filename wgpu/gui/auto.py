"""
Automatic GUI backend selection.
"""


def is_jupyter():
    return False


if is_jupyter():
    from .jupyter import WgpuCanvas, run, call_later  # noqa
else:
    from .glfw import WgpuCanvas, run, call_later  # noqa
