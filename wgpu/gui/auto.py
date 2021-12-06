"""
Automatic GUI backend selection.
"""


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
    from .glfw import WgpuCanvas, run, call_later  # noqa
