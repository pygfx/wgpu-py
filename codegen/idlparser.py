"""
The logic to parse the IDL file, from this we generate the base API.

This module may need tweaks as the used IDL syntax/constructs changes.

It would be good to occasionally check the coverage of this module to
identify and remove code paths that are no longer used.
"""

import os

from codegen.utils import print, lib_dir


_parser = None


def get_idl_parser(*, allow_cache=True):
    """Get the global IdlParser object."""

    # Singleton pattern
    global _parser
    if _parser and allow_cache:
        return _parser

    # Get source
    with open(os.path.join(lib_dir, "resources", "webgpu.idl"), "rb") as f:
        source = f.read().decode()

    # Create parser
    idl = IdlParser(source)
    idl.parse()
    _parser = idl
    return idl


class StructField:
    """A little object to specify the field of a struct."""

    def __init__(self, line, name, typename, default=None):
        self.line = line
        self.name = name
        self.typename = typename
        self.default = default

    def __repr__(self):
        return f"<StructField '{self.typename} {self.name}'>"

    def to_str(self):
        return self.line


class Interface:
    """A class definition, or flags."""

    def __init__(self, name, bases):
        self.bases = bases
        self.constants = {}
        self.attributes = {}  # name -> line
        self.functions = {}


class IdlParser:
    """An object that can be used to walk over a str in an easy way.

    This parser has the following attributes:

    * flags: a dict mapping the (neutral) flag name to a dict of field-value pairs.
    * enums: a dict mapping the (Pythonic) enum name to a dict of field-value pairs.
    * structs: a dict mapping the (Pythonic) struct name to a dict of StructField
      objects.
    * functions: a dict mapping the (normalized) func name to the line defining the
      function.

    """

    def __init__(self, source):

        self.source = self._pre_process(source)
        self._length = len(self.source)
        self._pos = 0

    def _reset(self):
        self._pos = 0

    def end_reached(self):
        return self._pos >= self._length

    def read_until(self, char):
        start = self._pos
        while self._pos < self._length:
            c = self.source[self._pos]
            self._pos += 1
            if c == char:
                return self.source[start : self._pos]
        return ""

    def read_line(self):
        return self.read_until("\n")

    def peek_line(self):
        char = "\n"
        start = pos = self._pos
        while pos < self._length:
            c = self.source[pos]
            pos += 1
            if c == char:
                return self.source[start:pos]
        return ""

    def parse(self, verbose=True):

        self._interfaces = {}
        self.classes = {}
        self.structs = {}
        self.flags = {}
        self.enums = {}

        self.typedefs = {}

        self._reset()
        self._parse()
        self._post_process()

        if verbose:
            f_count = sum(len(cls.functions) for cls in self.classes.values())
            print(
                f"The webgpu.idl defines {len(self.classes)} classes with {f_count} functions"
            )
            keys = "flags", "enums", "structs"
            stats = ", ".join(f"{len(getattr(self, key))} {key}" for key in keys)
            print("The webgpu.idl defines " + stats)

    def _pre_process(self, text):
        """Pre-process the text to make it a bit easier to parse.
        Beware to keep line numbers the same
        """
        text = text.replace("\n[\n", "\n\n[").replace("\n]\n", "]\n\n")
        text = text.replace("[    ", "[")
        text = self._remove_comments(text)
        return text

    def _remove_comments(self, text):
        lines = []
        in_multiline_comment = False
        for line in text.splitlines():
            if in_multiline_comment:
                if "*/" in line:
                    _, _, line = line.partition("//")
                    if "//" in line:
                        line, _, _ = line.partition("//")
                    lines.append(line if line.strip() else "")
                    in_multiline_comment = False
                else:
                    lines.append("")
            else:
                if "//" in line:
                    line, _, _ = line.partition("//")
                    lines.append(line if line.strip() else "")
                elif "/*" in line:
                    line, _, _ = line.partition("/*")
                    lines.append(line if line.strip() else "")
                    in_multiline_comment = True
                else:
                    lines.append(line)
        return "\n".join(lines)

    def resolve_type(self, typename):
        """Resolve a type to a suitable name that is also valid so that flake8
        wont complain when this is used as a type annotation.
        """

        name = typename.strip().strip("?")

        # We want the flag, not the type that is an alias for int
        name = name[:-5] if name.endswith("Flags") else name

        # First resolve using typedefs that we found in the IDL
        while name in self.typedefs:
            new_name = self.typedefs[name]
            if new_name == name:
                break
            name = new_name

        # Resolve to a Python type (maybe)
        pythonmap = {
            "DOMString": "str",
            "DOMString?": "str",
            "USVString": "str",
            "long": "int",
            "unsigned long": "int",
            "unsigned long long": "int",
            "[Clamp] unsigned short": "int",
            "unsigned short": "int",
            "GPUIntegerCoordinate": "int",
            "GPUSampleMask": "int",
            "GPUFenceValue": "int",
            "GPUSize64": "int",
            "GPUSize32": "int",
            "GPUIndex32": "int",
            "double": "float",
            "boolean": "bool",
            "object": "dict",
            "ImageBitmap": "memoryview",
        }
        name = pythonmap.get(name, name)

        # Is this a case for which we need to recurse?
        if name.startswith("sequence<") and name.endswith(">"):
            name = name.split("<")[-1].rstrip(">")
            name = self.resolve_type(name).strip("'")
            return f"'List[{name}]'"
        elif name.startswith("record<") and name.endswith(">"):
            name = name.split("<")[-1].rstrip(">")
            names = [self.resolve_type(t).strip("'") for t in name.split(",")]
            return f"'Dict[{', '.join(names)}]'"
        elif " or " in name:
            name = name.strip("()")
            names = [self.resolve_type(t).strip("'") for t in name.split(" or ")]
            return f"'Union[{', '.join(names)}]'"

        # Triage
        if name in __builtins__:
            return name  # ok
        elif name in self.classes:
            return f"'{name}'"  # ok, but wrap in string because can be declared later
        else:
            assert name.startswith("GPU")
            name = name[3:]
            name = name[:-4] if name.endswith("Dict") else name
            if name in self.flags:
                return f"'flags.{name}'"
            elif name in self.enums:
                return f"'enums.{name}'"
            elif name in self.structs:
                return f"'structs.{name}'"
            else:
                # When this happens, update the code above or the pythonmap
                raise RuntimeError("Encountered unknown IDL type: ", name)

    def _parse(self):

        while not self.end_reached():

            line = self.read_line()

            if not line.strip():
                pass
            elif line.startswith("typedef "):
                # Get the important bit
                value = line.split(" ", 1)[-1]
                if value.startswith("["):
                    value = value.split("]")[-1]
                # Parse
                if value.startswith("("):  # Union type
                    assert value.count("(") == 1 and value.count(")") == 1
                    value = value.split("(")[1]
                    val, _, key = value.partition(")")
                else:  # Singleton type
                    val, _, key = value.rpartition(" ")
                key = key.strip().strip(";").strip()
                self.typedefs[key] = val.strip()
            elif line.startswith(("interface ", "partial interface ")):
                # A class or a set of flags
                # Collect lines that define this interface
                lines = [line]
                while not line.startswith("};"):
                    line = self.read_line()
                    lines.append(line)
                classname = lines[0].split("{")[0].split(":")[0].split()[-1]
                # Collect base classes
                based_on = []
                while self.peek_line().startswith(classname + " includes "):
                    line = self.read_line()
                    based_on.append(line.split()[-1].rstrip(";"))
                # Create / get interface object
                if classname not in self._interfaces:
                    self._interfaces[classname] = Interface(classname, based_on)
                interface = self._interfaces[classname]
                # Parse members
                line_index = 0
                while line_index < len(lines) - 1:
                    line_index += 1
                    line = lines[line_index].strip()
                    if not line:
                        continue
                    elif line.startswith("[Exposed="):
                        continue  # WTF?
                    elif line.startswith("const "):
                        parts = line.strip(";").split()
                        assert len(parts) == 5
                        assert parts[-2] == "="
                        name = parts[2]
                        val = int(parts[-1], 16)
                        interface.constants[name] = val
                    elif "attribute " in line:
                        name = line.partition("attribute")[2].split()[-1].strip(";")
                        interface.attributes[name] = line
                    elif "(" in line:
                        line = lines[line_index]
                        while line.count("(") > line.count(")"):
                            line_index += 1
                            line += lines[line_index]
                        assert line.count("(") == line.count(")")
                        line = line.strip()
                        line.replace("\n", " ")
                        for c in ("    ", "  ", "  "):
                            line = line.replace(c, " ")
                        assert line.endswith(";")
                        funcname = line.split("(")[0].split()[-1]
                        line = (
                            line.replace("\n", " ")
                            .replace("    ", " ")
                            .replace("  ", " ")
                        )
                        interface.functions[funcname] = line
            elif " includes " in line:
                parts = line.strip(";").split()
                assert len(parts) == 3 and parts[1] == "includes"
                classname, _, base = parts
                if classname not in self._interfaces:
                    self._interfaces[classname] = Interface(classname, [])
                self._interfaces[classname].bases.append(parts[2])
            elif line.startswith("enum "):
                line += self.read_until("}") + self.read_line()
                lines = line.strip().split("\n")
                name = lines[0].split(" ", 1)[1].strip("{ \t\r\n")
                d = {}
                for i, line in enumerate(lines[1:-1]):
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    key = val = line.strip('", \t')
                    for i1, i2 in [
                        ("-", "_"),
                        ("1d", "d1"),
                        ("2d", "d2"),
                        ("3d", "d3"),
                    ]:
                        key = key.replace(i1, i2)
                    d[key] = val
                self.enums[name] = d
            elif line.startswith("dictionary "):
                assert line.count("{") == 1 and line.count("}") == 0
                lines = [line]
                while not line.startswith("};"):
                    line = self.read_line()
                    lines.append(line)
                name = lines[0].split(" ", 1)[1].strip("{ \t\r\n")
                if ":" in name:
                    name, _, base = name.partition(":")
                    name, base = name.strip(), base.strip()
                    if base not in self.structs:
                        # print(f"dict {name} has unknown base dict {base}")
                        d = {}
                    else:
                        d = self.structs[base].copy()
                else:
                    d = {}
                for line in lines[1:-1]:
                    line = line.split("//")[0].strip()
                    if not line:
                        continue
                    assert line.endswith(";")
                    arg = line.strip().strip(",;").strip()
                    default = None
                    if "=" in arg:
                        arg, default = arg.rsplit("=", 1)
                        arg, default = arg.strip(), default.strip()
                    arg_type, arg_name = arg.strip().rsplit(" ", 1)
                    if arg_type.startswith("required "):
                        arg_type = arg_type[9:]
                        # required args should not have a default
                        assert default is None
                    else:
                        default = default or "None"
                    d[arg_name] = StructField(line, arg_name, arg_type, default)
                self.structs[name] = d
            elif line.startswith(("[Exposed=", "[Serializable]")):
                pass
            else:
                raise RuntimeError("Unknown line:", line.rstrip())

    def _post_process(self):
        """We don't do any name format normalization in the parser code itself;
        we do that here.
        """

        # Drop some toplevel names
        for name in ["NavigatorGPU", "GPUAdapterLimits", "GPUSupportedFeatures"]:
            self._interfaces.pop(name, None)

        # Divide flags and actual class definitions
        for name, interface in self._interfaces.items():
            if interface.constants:
                self.flags[name] = interface.constants
            elif name not in ("Navigator", "WorkerNavigator"):
                delattr(interface, "constants")
                self.classes[name] = interface

        # Remove GPU prefix
        for d in (self.structs, self.flags, self.enums):
            for name in list(d.keys()):
                assert name.startswith("GPU")
                new_name = name[3:]
                if new_name.endswith("Dict"):
                    new_name = new_name[:-4]
                d[new_name] = d.pop(name)

        # Remove (abstract) base structs
        for name in list(self.structs):
            if name.endswith("Base"):
                self.structs.pop(name)
