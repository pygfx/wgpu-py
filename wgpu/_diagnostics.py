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

    def print_report(self):
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

    def print_report(self):
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
                    extrarows = [[""] * (i - 1) + subrow for subrow in subrows[1:]]
                    rows.extend(extrarows)
                break  # header items are consumed by the sub
            else:
                raise TypeError(f"Unexpected table value: {val}")

    return rows


# Root object
diagnostics = Diagnostics()
