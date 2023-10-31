class Diagnostics:
    """Base object to access wgpu diagnostics."""

    def __init__(self):
        # Init variables for the backends. As the BackendDiagnostics are created,
        # these are overwritten.
        self.base = None
        self.rs = None

    def get_report(self):
        """Get the full textual diagnostic report.

        See e.g. ``wgpu.diagnostics.rs.get_report_dict()`` for a report that
        can be processed programmatically.
        """
        text = ""
        for key, ob in self.__dict__.items():
            if not key.startswith("_") and isinstance(ob, BackendDiagnostics):
                d = ob.get_report_dict()
                if d:
                    text += ob.get_report()
        return text

    def print_report(self):  # no-cover
        """Print the full diagnostics report."""
        print(self.get_report())


class BackendDiagnostics:
    """Object to manage diagnostics for a specific wgpu backend."""

    def __init__(self, name):
        setattr(diagnostics, name, self)
        self.name = name
        self.object_counts = {}
        self.tracker = ObjectTracker(self.object_counts)

    def get_report_header(self):
        """Get the header for the report. These match the keys in the report dict."""
        return ("", "#py")

    def get_report_dict(self):
        """Get the diagnostics report for this backend, in the form of a Python dict."""
        # The default just shows the Python counts
        report = {}
        for name in sorted(self.object_counts.keys()):
            count = self.object_counts[name]
            report[name] = {"#py": count}
        return report

    def get_report(self):
        """Get the textual diagnostics report for this backend."""
        text = f"\nDiagnostics for wgpu - {self.name} backend:\n\n"
        text += dict_to_text(self.get_report_dict(), self.get_report_header())
        return text

    def print_report(self):  # no-cover
        """Print the diagnostics report for this backend."""
        print(self.get_report())


class ObjectTracker:
    """Little object to help track object counts."""

    def __init__(self, counts):
        self._counts = counts

    def increase(self, cls):
        name = cls.__name__[3:]
        self._counts[name] = self._counts.get(name, 0) + 1

    def decrease(self, cls):
        name = cls.__name__[3:]
        self._counts[name] -= 1


def dict_to_text(d, header):
    """Convert a dict data structure to a textual table representation."""

    # Get a table
    rows = dict_to_table(d, header)
    ncols = len(header)

    # Establish max lengths
    max_lens = [len(key) for key in header]
    for row in rows:
        assert len(row) == ncols, "dict_to_table failed"
        for i in range(ncols):
            assert isinstance(row[i], str), "dict_to_table failed"
            max_lens[i] = max(max_lens[i], len(row[i]))

    # Justify
    for row in rows:
        for i in range(ncols):
            row[i] = row[i].rjust(max_lens[i])

    # Join, with header
    first_line = "  ".join(header[i].rjust(max_lens[i]) for i in range(ncols))
    lines = [first_line, ""] + ["  ".join(row) for row in rows]
    lines.append("")  # end with empty line

    return "\n".join(line.rstrip() for line in lines)


def dict_to_table(d, header, header_offest=0):
    """Convert a dict data structure to a table (a list if lists of strings).
    The keys form the first entry of the row. Values that are dicts recurse.
    """

    ncols = len(header)
    rows = []

    for row_title, values in d.items():
        row = [row_title]
        rows.append(row)
        for i in range(header_offest + 1, len(header)):
            key = header[i]
            val = values.get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.6g}")
            elif isinstance(val, (int, str)):
                row.append(str(val))
            elif isinstance(val, dict):
                subrows = dict_to_table(val, header, i)
                if len(subrows) == 0:
                    while len(row) < ncols:
                        row.append("")
                else:
                    row += subrows[0]
                    extrarows = [[""] * i + subrow for subrow in subrows[1:]]
                    rows.extend(extrarows)
                break  # header items are consumed by the sub
            else:  # no-cover
                raise TypeError(f"Unexpected table value: {val}")

    return rows


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
    "rgb9e5ufloat": 32, # 3*9 + 5
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
    # Also see https://wgpu.rs/doc/wgpu/enum.TextureFormat.html
    "bc1-rgba-unorm": 4,  # 4x4 blocks, 8 bytes per block
    "bc1-rgba-unorm-srgb": 4,
    "bc2-rgba-unorm": 8,   # 4x4 blocks, 16 bytes per block
    "bc2-rgba-unorm-srgb": 8,
    "bc3-rgba-unorm": 8,  #4x4 blocks, 16 bytes per block
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
    # astc always uses 16 bytes (128 bits) per block.
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


# Root object
diagnostics = Diagnostics()
