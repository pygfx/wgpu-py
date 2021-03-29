"""
Core utilities that are loaded into the root namespace or used internally.
"""

import os
import inspect
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
            d[f.__qualname__] = text
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


# todo: remove or revive part of this for the rs codegen/maintenance
def _help(*searches, dev=False):
    """Print constants, enums, structs, and functions that contain the given searches.
    If dev is True, will also print info from the definitions in .idl and .h, which
    can be useful during debugging and dev.
    """

    from . import base as m_classes, flags as m_flags, enums as m_enums
    from ._parsers import IdlParser, HParser, to_neutral_name

    # Strip prefixes used in .idl and .h
    name_parts = []
    for name_part in searches:
        if name_part.lower().startswith("wgpu"):
            name_part = name_part[4:]
        elif name_part.lower().startswith("gpu"):
            name_part = name_part[3:]
        name_parts.append(name_part)

    # Prepare
    quoted = [f"{name_part!r}" for name_part in name_parts]
    print(f"Searching for {' and '.join(quoted)} ...")
    name_parts = [name_part.lower() for name_part in name_parts]
    all_lines = []  # list of (title, lines_list)

    if dev:
        filename = os.path.join(get_resource_filename("webgpu.idl"))
        idl_parser = IdlParser(open(filename, "rb").read().decode())
        idl_parser.parse()
        filename = os.path.join(get_resource_filename("wgpu.h"))
        h_parser = HParser(open(filename, "rb").read().decode())
        h_parser.parse()

    # Find flags
    lines = []
    all_lines.append(("flags", lines))
    for name_part in name_parts:
        for name, val in m_flags.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(val, m_flags.Flags):
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(val))
                else:
                    for option in val:
                        if name_part in option.lower():
                            lines.append(name + "." + option)

    if dev:
        lines = []
        all_lines.append(("flags in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.flags.items():
                if name_part in name.lower():
                    lines.append(
                        name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys())
                    )
                else:
                    for key in d.keys():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {d[key]}")
        lines = []
        all_lines.append(("flags in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.flags.items():
                if name_part in name.lower():
                    lines.append(
                        name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys())
                    )
                else:
                    for key in d.keys():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {d[key]}")

    # Find enums
    lines = []
    all_lines.append(("enums", lines))
    for name_part in name_parts:
        for name, val in m_enums.__dict__.items():
            if name.startswith("_"):
                continue
            elif isinstance(val, m_enums.Enum):
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"'{x}'" for x in val))
                else:
                    for option in val:
                        if name_part in option.lower():
                            lines.append(name + "." + option)

    if dev:
        lines = []
        all_lines.append(("enums in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.enums.items():
                if name_part in name.lower():
                    lines.append(
                        name + ": " + ", ".join(f"{key}='{d[key]}'" for key in d.keys())
                    )
                else:
                    for key, val in d.items():
                        if name_part in key.lower() or name_part in val.lower():
                            lines.append(f"{name}.{key} = '{val}'")
        lines = []
        all_lines.append(("enums in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.enums.items():
                if name_part in name.lower():
                    lines.append(
                        name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys())
                    )
                else:
                    for key, val in d.items():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {val}")

    # Find functions
    lines = []
    all_lines.append(("functions", lines))
    for name_part in name_parts:
        name_part_f = to_neutral_name(name_part)
        for cls in m_classes.__dict__.values():
            if isinstance(cls, type):
                for attr_name, attr in cls.__dict__.items():
                    if attr_name.startswith("_") or not callable(attr):
                        continue
                    funcname = cls.__name__ + "." + attr_name
                    sig = (
                        str(inspect.signature(attr))
                        .replace("self, ", "")
                        .replace("(self)", "()")
                    )
                    func_id = to_neutral_name(funcname)
                    if name_part_f in func_id or name_part_f in sig.lower():
                        lines.append(funcname + sig)

    if dev:
        lines = []
        all_lines.append(("functions in .idl", lines))
        for name_part in name_parts:
            name_part_f = name_part.replace("_", "").replace(".", "")
            for func_id, line in idl_parser.functions.items():
                if name_part_f in func_id:
                    lines.append(line.strip())
        lines = []
        all_lines.append(("functions in .h", lines))
        for name_part in name_parts:
            name_part_f = name_part.replace("_", "").replace(".", "")
            for func_id, line in h_parser.functions.items():
                if name_part_f in func_id:
                    lines.append(line.strip())

    # Find structs

    if dev:
        lines = []
        all_lines.append(("structs in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.structs.items():
                if name_part in name.lower():
                    x = "GPU" + name + " {"
                    for field in d.values():
                        x += "\n        " + field.line
                    x += "\n    }"
                    lines.append(x)
        lines = []
        all_lines.append(("structs in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.structs.items():
                if name_part in name.lower():
                    x = "WGPU" + name + " {"
                    for field in d.values():
                        x += "\n        " + field.line
                    x += "\n    }"
                    lines.append(x)

    # Display
    for title, lines in all_lines:
        # Remove duplicates
        for i in reversed(range(len(lines))):
            if lines[i] in lines[:i]:
                lines.pop(i)
        # Print header
        if " in " not in title:
            print(f"\n--- {len(lines)} {title} ".ljust(80, "-"))
        elif lines:
            print(f"--- {len(lines)} {title} ---")
        # Print lines
        if lines:
            print("\n".join("    " + line for line in lines))
