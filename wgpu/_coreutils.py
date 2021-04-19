"""
Core utilities that are loaded into the root namespace or used internally.
"""

import logging
from pkg_resources import resource_filename


def get_resource_filename(name):
    """Get the filename to a wgpu resource."""
    return resource_filename("wgpu.resources", name)


class WGPULogger(logging.getLoggerClass()):
    """A custom logger for which we can detect changes in its level."""

    def setLevel(self, level):  # noqa: N802
        super().setLevel(level)
        for cb in logger_set_level_callbacks:
            cb(self.level)  # use arg that is always an int


logger_set_level_callbacks = []
_original_logger_cls = logging.getLoggerClass()
logging.setLoggerClass(WGPULogger)
logger = logging.getLogger("wgpu")
logging.setLoggerClass(_original_logger_cls)
assert isinstance(logger, WGPULogger)
logger.setLevel(logging.WARNING)


class ApiDiff:
    """Helper class to define differences in the API by annotating
    methods. This way, these difference are made explicit, plus they're
    logged so we can automatically included these changes in the docs.
    """

    def __init__(self):
        self.hidden = {}
        self.added = {}
        self.changed = {}

    def hide(self, func_or_text):
        """Decorator to discard certain methods from the "reference" API.
        Intended only for the base API where we deviate from WebGPU.
        """
        return self._diff("hidden", func_or_text)

    def add(self, func_or_text):
        """Decorator to add certain methods that are not part of the "reference" spec.
        Intended for the base API where we implement additional/alternative API,
        and in the backend implementations where additional methods are provided.
        """
        return self._diff("added", func_or_text)

    def change(self, func_or_text):
        """Decorator to mark certain methods as having a different signature
        as the "reference" spec. Intended only for the base API where we deviate
        from WebGPU.
        """
        return self._diff("changed", func_or_text)

    def _diff(self, method, func_or_text):
        def wrapper(f):
            d = getattr(self, method)
            name = f.__qualname__ if hasattr(f, "__qualname__") else f.fget.__qualname__
            d[name] = text
            return f

        if callable(func_or_text):
            text = None
            return wrapper(func_or_text)
        else:
            text = func_or_text
            return wrapper

    def remove_hidden_methods(self, scope):
        """Call this to remove methods from the API that were decorated as hidden."""
        for name in self.hidden:
            classname, _, methodname = name.partition(".")
            cls = scope[classname]
            delattr(cls, methodname)

    @property
    def __doc__(self):
        """Generate a docstring for this instance. This way we can
        automatically document API differences.
        """
        lines = [""]
        for name, msg in self.hidden.items():
            line = f"    * Hides ``{name}()``"
            lines.append(f"{line} - {msg}" if msg else line)
        for name, msg in self.added.items():
            line = f"    * Adds ``{name}()``"
            lines.append(f"{line} - {msg}" if msg else line)
        for name, msg in self.changed.items():
            line = f"    * Changes ``{name}()``"
            lines.append(f"{line} - {msg}" if msg else line)
        lines.append("")
        return "\n".join(sorted(lines))
