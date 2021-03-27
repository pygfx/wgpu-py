"""
The logic to parse the IDL file, from this, we generate the API.
"""

from .utils import to_snake_case


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

    def py_arg(self):
        name = to_snake_case(self.name)
        t = self.typename
        d = self.default
        if t not in ("bool", "int", "float", "str"):
            t = f"'{t}'"
        if d is not None:
            d = {"false": "False", "true": "True"}.get(d, d)
            return f"{name}: {t}={d}"
        else:
            return f"{name}: {t}"


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

    def __init__(self, text):
        self._text = self._pre_process(text)
        self._length = len(self._text)
        self._pos = 0

    def _reset(self):
        self._pos = 0

    def end_reached(self):
        return self._pos >= self._length

    def read_until(self, char):
        start = self._pos
        while self._pos < self._length:
            c = self._text[self._pos]
            self._pos += 1
            if c == char:
                return self._text[start : self._pos]
        return ""

    def read_line(self):
        return self.read_until("\n")

    def peek_line(self):
        char = "\n"
        start = pos = self._pos
        while pos < self._length:
            c = self._text[pos]
            pos += 1
            if c == char:
                return self._text[start:pos]
        return ""

    def parse(self, verbose=False):

        self._interfaces = {}
        self.classes = {}
        self.structs = {}
        self.flags = {}
        self.enums = {}

        if verbose:
            print("##### Parsing IDL ...")

        self._reset()
        self._parse()
        self._post_process()

        if verbose:
            func_count = sum(len(cls.functions) for cls in self.classes.values())
            print(f"Found {len(self.classes)} classes with {func_count} functions")
            for thing in ["structs", "flags", "enums"]:
                print(f"Found {len(getattr(self, thing))} {thing}")

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

    def _parse(self):

        typedefs = {}

        while not self.end_reached():

            line = self.read_line()

            if not line.strip():
                pass
            elif line.startswith("typedef "):
                pass
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
                    arg_type = typedefs.get(arg_type, arg_type)
                    if arg_type in ["double", "float"]:
                        t = "float"
                    elif arg_type in [
                        "long",
                        "unsigned long",
                        "unsigned long long",
                        "GPUSize64",
                        "[Clamp] unsigned short",
                    ]:
                        t = "int"
                    elif arg_type in ["boolean"]:
                        t = "bool"
                    elif arg_type in ["DOMString", "DOMString?", "USVString"]:
                        t = "str"
                    elif arg_type in ["object", "record<DOMString, GPUSize32>"]:
                        t = "dict"
                    elif arg_type.startswith("GPU"):
                        t = arg_type
                    elif arg_type.startswith("sequence<GPU"):
                        t = arg_type[9:-1] + "-list"
                    elif arg_type == "ImageBitmap":
                        t = "array"
                    elif arg_type in [
                        "(GPULoadOp or GPUColor)",
                        "(GPULoadOp or GPUStencilValue)",
                        "(GPULoadOp or float)",
                        "(GPULoadOp or unsigned long)",
                    ]:
                        # GPURenderPassColorAttachmentDescriptor
                        # GPURenderPassDepthStencilAttachmentDescriptor
                        t = (
                            arg_type[1:-1]
                            .replace(" ", "-")
                            .replace("unsigned-long", "int")
                        )
                    else:
                        assert False
                    d[arg_name] = StructField(line, arg_name, t, default)
                self.structs[name] = d
            elif line.startswith(("[Exposed=", "[Serializable]")):
                pass
            else:
                raise RuntimeError("Unknown line:", line.rstrip())

    def _post_process(self):
        """We don't do any name format normalization in the parser code itself;
        we do that here.
        """

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

        # Turn funcs into python name
        for cls in self.classes.values():
            for funcname in list(cls.functions.keys()):
                funcname2 = to_snake_case(funcname)
                cls.functions[funcname2] = cls.functions.pop(funcname)
