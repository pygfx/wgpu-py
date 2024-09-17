"""
Logic related to providing diagnostic info on wgpu.
"""

import os
import sys
import platform


class DiagnosticsRoot:
    """Root object to access wgpu diagnostics (i.e. ``wgpu.diagnostics``).

    Per-topic diagnostics can be accessed as attributes on this object.
    These include ``system``, ``wgpu_native_info``, ``versions``,
    ``object_counts``, ``wgpu_natrive_counts``.
    """

    def __init__(self):
        self._diagnostics_instances = {}

    def __repr__(self):
        topics = ", ".join(self._diagnostics_instances.keys())
        return f"<DiagnosticsRoot with topics: {topics}>"

    def _register_diagnostics(self, name, ob):
        self._diagnostics_instances[name] = ob
        setattr(self, name, ob)

    def get_dict(self):
        """Get a dict that represents the full diagnostics info.

        The keys are the diagnostic topics, and the values are dicts
        of dicts. See e.g. ``wgpu.diagnostics.counts.get_dict()`` for
        a topic-specific dict.
        """
        result = {}
        for name, ob in self._diagnostics_instances.items():
            result[name] = ob.get_dict()
        return result

    def get_report(self):
        """Get the full textual diagnostic report (as a str)."""
        text = ""
        for name, ob in self._diagnostics_instances.items():
            text += ob.get_report()
        return text

    def print_report(self):
        """Convenience method to print the full diagnostics report."""
        print(self.get_report(), end="")


class DiagnosticsBase:
    """Object that represents diagnostics on a specific topic.

    This is a base class that must be subclassed to provide diagnostics on
    a certain topic. Typically only ``get_dict()`` needs to be implemented.
    Instantiating the class registers it with the root diagnostics object.
    """

    def __init__(self, name):
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError(
                "Diagnostics name must be an identifier (i.e. use underscore instead of spaces)."
            )
        self.name = name
        diagnostics._register_diagnostics(name, self)

    def __repr__(self):
        return f"<Diagnostics for '{self.name}'>"

    def get_dict(self):
        """Get the diagnostics for this topic, in the form of a Python dict.

        Subclasses must implement this method. The dict can be a simple
        map of keys to values (str, int, float)::

            foo: 1
            bar: 2

        If the values are dicts, the data has a table-like layout, with
        the keys representing the table header::

                      count  mem

            Adapter:      1  264
             Buffer:      4  704

        Subdicts are also supported, which results in multi-row entries.
        In the report, the keys of the subdicts have colons behind them::

                      count  mem  backend  o  v  e  el_size

            Adapter:      1  264  vulkan:  1  0  0      264
                                   d3d12:  1  0  0      220
             Buffer:      4  704  vulkan:  4  0  0      176
                                   d3d12:  0  0  0      154

        """
        raise NotImplementedError()

    def get_subscript(self):
        """Get informative text that helps interpret the report.

        Subclasses can implement this method. The text will show below the table
        in the report.
        """
        return ""  # Optional

    def get_report(self):
        """Get the textual diagnostics report for this topic."""
        text = f"\n██ {self.name}:\n\n"
        text += dict_to_text(self.get_dict())
        subscript = self.get_subscript()
        if subscript:
            text += "\n" + subscript.rstrip() + "\n"
        return text

    def print_report(self):
        """Print the diagnostics report for this topic."""
        print(self.get_report(), end="")


class ObjectTracker:
    """Little object to help track object counts."""

    def __init__(self):
        self.counts = {}
        self.amounts = {}

    def increase(self, name, amount=0):
        """Bump the counter."""
        self.counts[name] = self.counts.get(name, 0) + 1
        if amount:
            self.amounts[name] = self.amounts.get(name, 0) + amount

    def decrease(self, name, amount=0):
        """Bump the counter back."""
        self.counts[name] -= 1
        if amount:
            self.amounts[name] -= amount


def derive_header(dct):
    """Derive a table-header from the given dict."""

    if not isinstance(dct, dict):  # no-cover
        raise TypeError(f"Not a dict: {dct}")

    header = []
    sub_dicts = {}

    for key, val in dct.items():
        if not isinstance(val, dict):  # no-cover
            raise TypeError(f"Element not a dict: {val}")
        for k, v in val.items():
            if k not in header:
                header.append(k)
            if isinstance(v, dict):
                sub_dicts[k] = v

    for k, d in sub_dicts.items():
        while k in header:
            header.remove(k)
        header.append(k)
        sub_header = derive_header(d)
        for k in sub_header[1:]:
            if k not in header:
                header.append(k)

    # Add header item for first column, i.e. the key / row title
    header.insert(0, "")

    return header


def dict_to_text(d, header=None):
    """Convert a dict data structure to a textual table representation."""

    if not d:
        return "No data\n"

    # Copy the dict, with simple key-value dicts being transformed into table-like dicts.
    # That wat the code in derive_header() and dict_to_table() can assume the table-like
    # data structure, keeping it simpler.
    d2 = {}
    for key, val in d.items():
        if not isinstance(val, dict):
            val = {"": val}
        d2[key] = val
    d = d2

    if not header:
        header = derive_header(d)

    # We have a table-like-layout if any of the values in the header is non-empty
    table_layout = any(header)

    # Get the table
    rows = dict_to_table(d, header)
    ncols = len(header)

    # Sanity check (guard assumptions about dict_to_table)
    for row in rows:
        assert len(row) == ncols, "dict_to_table failed"
        for i in range(ncols):
            assert isinstance(row[i], str), "dict_to_table failed"

    # Insert heading
    if table_layout:
        rows.insert(0, header.copy())
        rows.insert(1, [""] * ncols)

    # Determine what colons have values with a colon at the end
    column_has_colon = [False for _ in range(ncols)]
    for row in rows:
        for i in range(ncols):
            column_has_colon[i] |= row[i].endswith(":")

    # Align the values that don't have a colon at the end
    for row in rows:
        for i in range(ncols):
            word = row[i]
            if column_has_colon[i] and not word.endswith(":"):
                row[i] = word + " "

    # Establish max lengths
    max_lens = [0 for _ in range(ncols)]
    for row in rows:
        for i in range(ncols):
            max_lens[i] = max(max_lens[i], len(row[i]))

    # Justify first column (always rjust)
    for row in rows:
        row[0] = row[0].rjust(max_lens[0])

    # For the table layout we also rjust the other columns
    if table_layout:
        for row in rows:
            for i in range(1, ncols):
                row[i] = row[i].rjust(max_lens[i])

    # Join into a consistent text
    lines = ["  ".join(row).rstrip() for row in rows]
    text = "\n".join(lines)
    return text.rstrip() + "\n"


def dict_to_table(d, header, header_offset=0):
    """Convert a dict data structure to a table (a list of lists of strings).
    The keys form the first entry of the row. Values that are dicts recurse.
    """

    ncols = len(header)
    rows = []

    for row_title, values in d.items():
        if row_title == "total" and row_title == list(d.keys())[-1]:
            rows.append([""] * ncols)
        row = [row_title + ":" if row_title else ""]
        rows.append(row)
        for i in range(header_offset + 1, len(header)):
            key = header[i]
            val = values.get(key, None)
            if val is None:
                row.append("")
            elif isinstance(val, str):
                row.append(val)
            elif isinstance(val, bool):
                row.append("✓" if val else "-")
            elif isinstance(val, int):
                row.append(int_repr(val))
            elif isinstance(val, float):
                row.append(f"{val:.6g}")
            elif isinstance(val, dict):
                subrows = dict_to_table(val, header, i)
                if len(subrows) == 0:
                    row += [""] * (ncols - i)
                else:
                    row += subrows[0]
                    extrarows = [[""] * i + subrow for subrow in subrows[1:]]
                    rows.extend(extrarows)
                break  # header items are consumed by the sub
            else:  # no-cover
                raise TypeError(f"Unexpected table value: {val}")

    return rows


def int_repr(val):
    """Represent an integer using K and M suffixes."""
    prefix = "-" if val < 0 else ""
    suffix = ""
    val = abs(val)
    s = str(val)
    tail = 3 * ((len(s) - 1) // 3)
    if tail > 0:
        s1, s2 = s[:-tail], s[-tail:]
        n_decimals = max(0, 3 - len(s1))
        s = s1
        if n_decimals:
            s2 += "000"
            s = s1 + "." + s2[:n_decimals]
        if tail == 3:
            suffix = "K"
        elif tail == 6:
            suffix = "M"
        elif tail == 9:
            suffix = "G"
        else:
            suffix = f"E{tail}"
    return prefix + s + suffix


# Map that we need to calculate texture resource consumption.
# We need to keep this up-to-date as formats change, we have a unit test for this.
# Also see https://wgpu.rs/doc/wgpu/enum.TextureFormat.html

texture_format_to_bpp = {
    # 8 bit
    "r8unorm": 8,
    "r8snorm": 8,
    "r8uint": 8,
    "r8sint": 8,
    # 16 bit
    "r16uint": 16,
    "r16sint": 16,
    "r16float": 16,
    "rg8unorm": 16,
    "rg8snorm": 16,
    "rg8uint": 16,
    "rg8sint": 16,
    # 32 bit
    "r32uint": 32,
    "r32sint": 32,
    "r32float": 32,
    "rg16uint": 32,
    "rg16sint": 32,
    "rg16float": 32,
    "rgba8unorm": 32,
    "rgba8unorm-srgb": 32,
    "rgba8snorm": 32,
    "rgba8uint": 32,
    "rgba8sint": 32,
    "bgra8unorm": 32,
    "bgra8unorm-srgb": 32,
    # special fits
    "rgb9e5ufloat": 32,  # 3*9 + 5
    "rgb10a2uint": 32,  # 3*10 + 2
    "rgb10a2unorm": 32,  # 3*10 + 2
    "rg11b10ufloat": 32,  # 2*11 + 10
    # 64 bit
    "rg32uint": 64,
    "rg32sint": 64,
    "rg32float": 64,
    "rgba16uint": 64,
    "rgba16sint": 64,
    "rgba16float": 64,
    # 128 bit
    "rgba32uint": 128,
    "rgba32sint": 128,
    "rgba32float": 128,
    # depth and stencil
    "stencil8": 8,
    "depth16unorm": 16,
    "depth24plus": 24,  # "... at least 24 bit integer depth" ?
    "depth24plus-stencil8": 32,
    "depth32float": 32,
    "depth32float-stencil8": 40,
    # Compressed
    "bc1-rgba-unorm": 4,  # 4x4 blocks, 8 bytes per block
    "bc1-rgba-unorm-srgb": 4,
    "bc2-rgba-unorm": 8,  # 4x4 blocks, 16 bytes per block
    "bc2-rgba-unorm-srgb": 8,
    "bc3-rgba-unorm": 8,  # 4x4 blocks, 16 bytes per block
    "bc3-rgba-unorm-srgb": 8,
    "bc4-r-unorm": 4,
    "bc4-r-snorm": 4,
    "bc5-rg-unorm": 8,
    "bc5-rg-snorm": 8,
    "bc6h-rgb-ufloat": 8,
    "bc6h-rgb-float": 8,
    "bc7-rgba-unorm": 8,
    "bc7-rgba-unorm-srgb": 8,
    "etc2-rgb8unorm": 4,
    "etc2-rgb8unorm-srgb": 4,
    "etc2-rgb8a1unorm": 4,
    "etc2-rgb8a1unorm-srgb": 4,
    "etc2-rgba8unorm": 8,
    "etc2-rgba8unorm-srgb": 8,
    "eac-r11unorm": 4,
    "eac-r11snorm": 4,
    "eac-rg11unorm": 8,
    "eac-rg11snorm": 8,
    # astc always uses 16 bytes (128 bits) per block
    "astc-4x4-unorm": 8.0,
    "astc-4x4-unorm-srgb": 8.0,
    "astc-5x4-unorm": 6.4,
    "astc-5x4-unorm-srgb": 6.4,
    "astc-5x5-unorm": 5.12,
    "astc-5x5-unorm-srgb": 5.12,
    "astc-6x5-unorm": 4.267,
    "astc-6x5-unorm-srgb": 4.267,
    "astc-6x6-unorm": 3.556,
    "astc-6x6-unorm-srgb": 3.556,
    "astc-8x5-unorm": 3.2,
    "astc-8x5-unorm-srgb": 3.2,
    "astc-8x6-unorm": 2.667,
    "astc-8x6-unorm-srgb": 2.667,
    "astc-8x8-unorm": 2.0,
    "astc-8x8-unorm-srgb": 2.0,
    "astc-10x5-unorm": 2.56,
    "astc-10x5-unorm-srgb": 2.56,
    "astc-10x6-unorm": 2.133,
    "astc-10x6-unorm-srgb": 2.133,
    "astc-10x8-unorm": 1.6,
    "astc-10x8-unorm-srgb": 1.6,
    "astc-10x10-unorm": 1.28,
    "astc-10x10-unorm-srgb": 1.28,
    "astc-12x10-unorm": 1.067,
    "astc-12x10-unorm-srgb": 1.067,
    "astc-12x12-unorm": 0.8889,
    "astc-12x12-unorm-srgb": 0.8889,
}


# %% global diagnostics object, and builtin diagnostics


# The global root object
diagnostics = DiagnosticsRoot()


class SystemDiagnostics(DiagnosticsBase):
    """Provides basic system info."""

    def get_dict(self):
        return {
            "platform": platform.platform(),
            # "platform_version": platform.version(),  # can be quite long
            "python_implementation": platform.python_implementation(),
            "python": platform.python_version(),
        }


class WgpuNativeInfoDiagnostics(DiagnosticsBase):
    """Provides metadata about the wgpu-native backend."""

    def get_dict(self):
        # Get modules, or skip
        try:
            wgpu = sys.modules["wgpu"]
            wgpu_native = wgpu.backends.wgpu_native
        except (KeyError, AttributeError):  # no-cover
            return {}

        # Process lib path
        lib_path = wgpu_native.lib_path
        wgpu_path = os.path.dirname(wgpu.__file__)
        if lib_path.startswith(wgpu_path):
            lib_path = "." + os.path.sep + lib_path[len(wgpu_path) :].lstrip("/\\")

        return {
            "expected_version": wgpu_native.__version__,
            "lib_version": ".".join(str(i) for i in wgpu_native.lib_version_info),
            "lib_path": lib_path,
        }


class VersionDiagnostics(DiagnosticsBase):
    """Provides version numbers from relevant libraries."""

    def get_dict(self):
        core_libs = ["wgpu", "cffi"]
        qt_libs = ["PySide6", "PyQt6", "PySide2", "PyQt5"]
        gui_libs = qt_libs + ["glfw", "jupyter_rfb", "wx"]
        extra_libs = ["numpy", "pygfx", "pylinalg", "fastplotlib"]

        info = {}

        for libname in core_libs + gui_libs + extra_libs:
            try:
                ver = sys.modules[libname].__version__
            except (KeyError, AttributeError):
                pass
            else:
                info[libname] = str(ver)

        return info


class ObjectCountDiagnostics(DiagnosticsBase):
    """Provides object counts and resource consumption, used in _classes.py."""

    def __init__(self, name):
        super().__init__(name)
        self.tracker = ObjectTracker()

    def get_dict(self):
        """Get diagnostics as a dict."""
        object_counts = self.tracker.counts
        resource_mem = self.tracker.amounts

        # Collect counts
        result = {}
        for name in sorted(object_counts.keys()):
            d = {"count": object_counts[name]}
            if name in resource_mem:
                d["resource_mem"] = resource_mem[name]
            result[name[3:]] = d  # drop the 'GPU' from the name

        # Add totals
        totals = {}
        for key in ("count", "resource_mem"):
            totals[key] = sum(v.get(key, 0) for v in result.values())
        result["total"] = totals

        return result


SystemDiagnostics("system")
VersionDiagnostics("versions")
WgpuNativeInfoDiagnostics("wgpu_native_info")
ObjectCountDiagnostics("object_counts")
