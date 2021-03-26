import sys

import black


def to_neutral_name(name):
    """Convert a name to the neutral name with no capitals, underscores or dots.
    Used primarily to match function names of IDL and .h specs.
    """
    name = name.lower()
    if name.startswith("gpu"):
        name = name[3:]
    if name.startswith("wgpu"):
        name = name[4:]
    for c in " -_.":
        name = name.replace(c, "")
    return name


def to_python_name(name):
    """Convert someName and some_name to the Python flavor.
    To convert function names and function argument names.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0 and name2[-1] != "_":
            name2 += "_"
        name2 += c2
    return name2


def blacken(src, singleline=False):
    """Format the given src string using black. If singleline is True,
    all function signatures become single-line, so they can be parsed
    and updated.
    """
    # Normal black
    result = black.format_str(src, mode=black.FileMode())

    # Make defs single-line. You'd think that setting the line length
    # to a very high number would do the trick, but it does not.
    if singleline:
        lines1 = result.splitlines()
        lines2 = []
        in_sig = False
        comment = ""
        for line in lines1:
            if in_sig:
                # Handle comment
                line, _, c = line.partition("#")
                line = line.rstrip()
                c = c.strip()
                if c:
                    comment += " " + c.strip()
                # Detect end
                if line.endswith("):"):
                    in_sig = False
                # Compose line
                current_line = lines2[-1]
                if not current_line.endswith("("):
                    current_line += " "
                current_line += line.lstrip()
                # Finalize
                if not in_sig:
                    # Remove trailing spaces and commas
                    current_line = current_line.replace(" ):", "):")
                    current_line = current_line.replace(",):", "):")
                    # Add comment
                    if comment:
                        current_line += "  #" + comment
                        comment = ""
                lines2[-1] = current_line
            else:
                lines2.append(line)
                line_nc = line.split("#")[0].strip()
                if (
                    line_nc.startswith(("def ", "async def", "class "))
                    and "(" in line_nc
                ):
                    if not line_nc.endswith("):"):
                        in_sig = True
        lines2.append("")
        result = "\n".join(lines2)

    return result


class Patcher:
    """Class to help patch a Python module. Supports iterating (over
    lines, classes, functions), and applying diffs (replace, remove, insert).
    """

    def __init__(self, code=None):
        self._init(code)

    def _init(self, code):
        self.lines = []
        self._diffs = {}
        self._classes = {}
        if code:
            self.lines = blacken(code, True).splitlines()  # inf line length

    def remove_line(self, i):
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "remove"

    def insert_line(self, i, line):
        if i in self._diffs and self._diffs[i][1] == "insert":
            cur_line = self._diffs[i][2]
            self._diffs[i] = i, "insert", cur_line + "\n" + line
        else:
            assert i not in self._diffs, f"Line {i} already has a diff"
            self._diffs[i] = i, "insert", line

    def replace_line(self, i, line):
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "replace", line

    def dumps(self, format=True):
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
            else:
                raise ValueError(f"Unknown diff: {diff}")
        # Format
        text = "\n".join(lines)
        if format:
            text = blacken(text)
        return text

    def iter_lines(self, start_line=0):
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            yield line, i

    def iter_classes(self, start_line=0):
        current_class = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if sline and not sline.startswith("    "):
                if current_class:
                    yield current_class + (i - 1,)
                    current_class = None
            if line.startswith("class "):
                name = line.split(":")[0].split("(")[0].split()[-1]
                current_class = name, i
                # self._classes[current_class] = i, current_class, {}
        if current_class:
            yield current_class + (i - 1,)

    def iter_properties(self, start_line=0):
        return self._iter_props_and_methods(start_line, True)

    def iter_methods(self, start_line=0):
        return self._iter_props_and_methods(start_line, False)

    def _iter_props_and_methods(self, start_line, find_props):
        prop_mark = None
        current_def = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if sline and not sline.startswith("        "):
                if current_def:
                    yield current_def + (i - 1,)
                    current_def = None
                if not sline.startswith("    "):
                    break  # exit class
            if line.startswith(("    def ", "    async def ")):
                name = line.split("(")[0].split()[-1]
                if prop_mark and find_props:
                    current_def = name, prop_mark
                elif not prop_mark and not find_props:
                    current_def = name, i
            if line.startswith("    @property"):
                prop_mark = i
            elif sline and not sline.startswith("#"):
                prop_mark = None

        if current_def:
            yield current_def + (i - 1,)

    def parse_public_api(self):
        current_class = None
        for i, line in enumerate(self.lines):
            if line.startswith("class "):
                current_class = line.split(":")[0].split("(")[0].split()[-1]
                self._classes[current_class] = i, current_class, {}
            if line.lstrip().startswith(("def ", "async def")):
                indent = len(line) - len(line.lstrip())
                funcname = line.split("(")[0].split()[-1]
                if not funcname.startswith("_"):
                    if not self.lines[i - 1].lstrip().startswith("@property"):
                        func_id = funcname
                        funcname = to_python_name(funcname)
                        if indent:
                            func_id = current_class + "." + func_id
                        func_id = to_neutral_name(func_id)
                        api_functions[func_id] = funcname, i, indent
