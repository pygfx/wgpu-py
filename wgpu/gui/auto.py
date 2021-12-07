"""
Automatic GUI backend selection.

Right now we only chose between GLFW and Jupyter. We might add suport
for e.g. Qt later. Or we might decide to stick with these two.
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
    try:
        from .glfw import WgpuCanvas, run, call_later  # noqa
    except ImportError as err:
        raise ImportError(
            str(err) + "\n\n  Install glfw using e.g. ``pip install -U glfw``"
        )
