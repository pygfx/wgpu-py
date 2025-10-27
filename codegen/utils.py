"""
Codegen utils.
"""

import os
import sys
import tempfile
import subprocess


def to_snake_case(name, separator="_"):
    """Convert a name from camelCase to snake_case. Names that already are
    snake_case remain the same.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0:
            prev = name2[-1]
            if c2 == "d" and prev in "123":
                name2 = name2[:-1] + separator + prev
            elif prev != separator:
                name2 += separator
        name2 += c2
    return name2


def to_camel_case(name):
    """Convert a name from snake_case to camelCase. Names that already are
    camelCase remain the same.
    """
    is_capital = False
    name2 = ""
    for c in name:
        if c in "_-" and name2:
            is_capital = True
        elif is_capital:
            name2 += c.upper()
            is_capital = False
        else:
            name2 += c
    if name2.endswith(("1d", "2d", "3d")):
        name2 = name2[:-1] + "D"
    return name2


_file_objects_to_print_to = [sys.stdout]


def print(*args, **kwargs):
    """Report something will be printed and added to a file."""
    # __builtins__.print(*args, **kwargs)
    if args and not args[0].lstrip().startswith("#"):
        args = ("*", *args)
    for f in _file_objects_to_print_to:
        __builtins__["print"](*args, file=f, flush=True, **kwargs)


class PrintToFile:
    """Context manager to print to file."""

    def __init__(self, f):
        assert hasattr(f, "write")
        self.f = f

    def __enter__(self):
        _file_objects_to_print_to.append(self.f)

    def __exit__(self, type, value, tb):
        while self.f in _file_objects_to_print_to:
            _file_objects_to_print_to.remove(self.f)
        self.f.close()


def remove_c_comments(code):
    """Remove C-style comments from the given code."""
    pos = 0
    new_code = ""

    while True:
        # Find start of comment
        lookfor = None
        i1 = code.find("//", pos)
        i2 = code.find("/*", pos)
        if i1 >= 0:
            lookfor = "\n"
            comment_start = i1
        if i2 >= 0:
            if not (i1 >= 0 and i1 < i2):
                lookfor = "*/"
                comment_start = i2
        # Found a start?
        if not lookfor:
            new_code += code[pos:]
            break
        else:
            new_code += code[pos:comment_start]
        # Find the end
        comment_end = code.find(lookfor, comment_start + 2)
        if comment_end < 0:
            break
        if lookfor == "\n":
            pos = comment_end
        else:
            pos = comment_end + len(lookfor)
    return new_code


class FormatError(Exception):
    pass


def format_code(src, singleline=False):
    """Format the given src string. If singleline is True, all function
    signatures become single-line, so they can be parsed and updated.
    """
    line_length = 320 if singleline else 88
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--line-length",
        str(line_length),
        "--no-cache",
        "--stdin-filename",
        "tmp.py",
    ]

    p = subprocess.run(cmd, input=src.encode(), capture_output=True)

    if p.returncode:
        stdout = p.stdout.decode(errors="ignore")
        stderr = p.stderr.decode(errors="ignore")
        raise FormatError(
            f"Could not format source ({p.returncode}).\n\nstdout: "
            + stdout
            + "\n\nstderr: "
            + stderr
        )
    else:
        result = p.stdout.decode(errors="ignore")

    # Make defs single-line. You'd think that setting the line length
    # to a very high number would do the trick, but it does not.
    if singleline:
        result = _make_sigs_singline(result)

    return result


def _make_sigs_singline(code):
    lines1 = code.splitlines()
    lines2 = []

    sig_state = 0
    sig_brace_depth = 0
    sig_line = ""
    sig_comment = ""

    for line in lines1:
        # Check to enter in signature-retrieval mode
        if not sig_state:
            if line.lstrip().startswith(("def ", "async def", "class ")):
                sig_state = 1
                sig_line = ""
                sig_comment = ""
                sig_brace_depth = 0
                if line.lstrip().startswith("class ") and "(" not in line:
                    sig_state = 3  # search for closing colon directly
            else:
                lines2.append(line)
                continue

        # If we get here, we're in a signature

        # Handle comment
        line, _, c = line.partition("#")
        line = line.rstrip()
        c = c.strip()
        if c:
            sig_comment += " " + c.strip()

        if sig_state == 1:
            # Find the first opening brace
            i = line.find("(")
            if i >= 0:
                i += 1
                sig_brace_depth = 1
                sig_line += line[:i]
                line = line[i:]
                sig_state = 2

        line = line.lstrip()
        if sig_line.endswith(","):
            line = " " + line

        if sig_state == 2:
            # Resolve braces until we find the closing brace
            while True:
                i1 = line.find("(")
                i2 = line.find(")")
                if i1 >= 0 and i1 < i2:
                    i = i1 + 1
                    sig_brace_depth += 1
                    sig_line += line[:i]
                    line = line[i:]
                elif i2 >= 0:
                    i = i2 + 1
                    sig_brace_depth -= 1
                    sig_line += line[:i]
                    line = line[i:]
                else:
                    break
                if sig_brace_depth == 0:
                    sig_state = 3
                    break

        if sig_state == 3:
            # Find the closing colon
            i = line.find(":")
            if i >= 0:
                i += 1
                sig_line += line[:i]
                line = line[i:]
                # Finish the signature line
                sig_line = sig_line.replace(", )", ")")
                if sig_comment:
                    sig_line += "  #" + sig_comment
                lines2.append(sig_line)
                if line.strip():
                    lines2.append(" " * 12 + line.strip())
                # End the search
                sig_state = 0
                line = ""

        sig_line += line

    lines2.append("")
    return "\n".join(lines2)


class Patcher:
    """Class to help patch a Python module. Supports iterating (over
    lines, classes, properties, methods), and applying diffs (replace,
    remove, insert).
    """

    def __init__(self, code=None):
        self._init(code)

    def _init(self, code):
        """Subclasses can call this to reset the patcher."""
        self.lines = []
        self._diffs = {}
        self._classes = {}
        if code:
            self.lines = format_code(code, True).splitlines()  # inf line length

    def remove_line(self, i):
        """Remove the line at the given position. There must not have been
        an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "remove"

    def insert_line(self, i, line):
        """Insert a new line at the given position. It's ok if there
        has already been an insertion on line i, but there must not have been
        any other actions.
        """
        if i in self._diffs and self._diffs[i][1] == "insert":
            cur_line = self._diffs[i][2]
            self._diffs[i] = i, "insert", cur_line + "\n" + line
        else:
            assert i not in self._diffs, f"Line {i} already has a diff"
            self._diffs[i] = i, "insert", line

    def replace_line(self, i, line):
        """Replace the line at the given position with another line.
        There must not have been an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "replace", line

    def dumps(self, format=True):
        """Return the patched result as a string."""
        lines = self.lines.copy()
        # Apply diff
        diffs = sorted(self._diffs.values())
        for diff in reversed(diffs):
            if diff[1] == "remove":
                lines.pop(diff[0])
            elif diff[1] == "insert":
                lines.insert(diff[0], diff[2])
            elif diff[1] == "replace":
                lines[diff[0]] = diff[2]
            else:  # pragma: no cover
                raise ValueError(f"Unknown diff: {diff}")
        # Format
        text = "\n".join(lines)
        if format:
            try:
                text = format_code(text)
            except FormatError as err:  # pragma: no cover
                # If you get this error, it really helps to load the code
                # in an IDE to see where the error is. Let's help with that ...
                filename = os.path.join(tempfile.gettempdir(), "wgpu_patcher_fail.py")
                with open(filename, "wb") as f:
                    f.write(text.encode())
                err = str(err)
                err = err if len(err) < 78 else err[:77] + "…"
                raise RuntimeError(
                    f"It appears that the patcher has generated invalid Python:"
                    f"\n\n    {err}\n\n"
                    f'Wrote the generated (but unformatted) code to:\n\n  "{filename}"'
                ) from None

        return text

    def iter_lines(self, start_line=0):
        """Generator to iterate over the lines.
        Each iteration yields (line, linenr)
        """
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            yield line, i

    def iter_classes(self, start_line=0):
        """Generator to iterate over the classes.
        Each iteration yields (classname, linenr_start, linenr_end),
        where linenr_end is the last line of code.
        """
        current_class = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_class and sline:
                if sline.startswith("    "):
                    current_class[2] = i
                else:  # code has less indentation -> something new
                    yield current_class
                    current_class = None
            if line.startswith("class "):
                name = line.split(":")[0].split("(")[0].split()[-1]
                current_class = [name, i, i]
        if current_class:
            yield current_class

    def iter_properties(self, start_line=0):
        """Generator to iterate over the properties.
        Each iteration yields (propertyname, linenr_first, linenr_last),
        where linenr_first is the line that startswith `def`,
        and linenr_last is the last line of code.
        """
        return self._iter_props_and_methods(start_line, True)

    def iter_methods(self, start_line=0):
        """Generator to iterate over the methods.
        Each iteration yields (methodname, linenr_first, linenr_last)
        where linenr_first is the line that startswith `def`,
        and linenr_last is the last line of code.
        """
        return self._iter_props_and_methods(start_line, False)

    def _iter_props_and_methods(self, start_line, find_props):
        prop_mark = None
        current_def = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_def and sline:
                if sline.startswith("        "):
                    current_def[2] = i
                else:
                    yield current_def
                    current_def = None
            if sline and not sline.startswith("    "):
                break  # exit class
            if line.startswith(("    def ", "    async def ")):
                name = line.split("(")[0].split()[-1]
                if prop_mark and find_props:
                    current_def = [name, i, i]
                elif not prop_mark and not find_props:
                    current_def = [name, i, i]
            if line.startswith("    @property"):
                prop_mark = i
            elif sline and not sline.lstrip().startswith("#"):
                prop_mark = None

        if current_def:
            yield current_def
