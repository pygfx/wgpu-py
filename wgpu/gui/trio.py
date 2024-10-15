"""A Trio-based event loop."""

import trio  # noqa
from .base import WgpuLoop


# todo: this would be nice
class TrioWgpuLoop(WgpuLoop):
    pass
