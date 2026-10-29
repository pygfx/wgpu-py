from typing import Literal

from apigen_utils import (
    dict_deep_merge,
    read_definition_file,
    read_file,
    write_definition_file,
)

builtins = __builtins__
if not isinstance(builtins, dict):
    builtins = builtins.__dict__


# Minimal difference applied to the final structure. Intended for changes that would otherwise make the API invalid Python.
diff_webgpu = {
    "classes": {
        "GPUPipelineError": {
            "methods": {
                # Make GPUPipelineError.__init__() message arg not have a default,
                # because the second arg has no default either
                "__init__": {"args": {"message": {"default": ""}}},
            }
        }
    }
}

# TODO: handle ARGS_TO_MAKE_OPTIONAL, in apidiff structure ??


def extract_idl():
    """Parse the IDL specification to a structured form in json.

    This is done in three stages:

    - Parsing the IDL and buil up an internal representation.
    - Convert that representation to our structured API dict form.
    - Resolve names, converting camelCase to snake_case, for struct fields, methods, and method arguments.
    - Resolve JS types to Python types.
    """

    # Get the source, and parse into a structure.
    source = read_file("apigen", "definitions", "webgpu.idl")
    parser = IdlParser(source)
    parser.parse()

    # Generate structure
    fname = "webgpu.json"
    api = {}
    api["meta"] = {
        "filename": fname,
        "description": "Python-flavored WebGPU API, generated from 'webgpu.idl'.",
        "notes": "All types are converted to the Python equivalents, all names are made Pythonic snake_case, methods that accept a single descriptor arg are *not* yet resolved to accept the descriptor args directly.",
    }
    api.update(build_api_dict(parser))

    # Convert names and types
    make_names_pythonic(api)
    resolve_types(api, parser)

    # Some final tweaks
    api = dict_deep_merge(api, diff_webgpu)

    # Write (and read back as a validation step)
    write_definition_file(api, fname)
    read_definition_file(fname)


def build_api_dict(parser):
    """Build api dict from the parser's result."""
    api = {}
    flags_dict = api["flags"] = {}
    for flag_name, d in parser.flags.items():
        flag_dict = flags_dict[flag_name] = {}
        flag_dict["fields"] = {}
        for k, v in d.items():
            flag_dict["fields"][k] = {"value": v}

    enums_dict = api["enums"] = {}
    for enum_name, d in parser.enums.items():
        enum_dict = enums_dict[enum_name] = {}
        enum_dict["fields"] = {}
        for k, v in d.items():
            enum_dict["fields"][k] = {"value": v}

    structs_dict = api["structs"] = {}
    for struct_name, d in parser.structs.items():
        struct_dict = structs_dict[struct_name] = {}
        struct_dict["fields"] = {}
        for att in d.values():
            struct_dict["fields"][att.name] = att.to_dict()

    classes_dict = api["classes"] = {}
    for class_name, interface in parser.classes.items():
        class_dict = classes_dict[class_name] = {}
        class_dict["bases"] = interface.bases
        class_dict["properties"] = {}
        for prop_name, attr_type in interface.properties.items():
            class_dict["properties"][prop_name] = {"type": attr_type}  # str
        class_dict["methods"] = {}
        for func in interface.methods.values():
            method_dict = class_dict["methods"][func.name] = {}
            method_dict["returns"] = {"type": func.return_type}
            method_dict["args"] = {}
            for attribute in func.args:
                method_dict["args"][attribute.name] = attribute.to_dict()

    return api


def make_names_pythonic(api):
    """Convert camelCase to snake_case."""

    # Struct field names
    for _struct_name, struct_dict in api["structs"].items():
        new_fields = {}
        for field_name, field_dict in struct_dict["fields"].items():
            new_fields[to_snake_case(field_name)] = field_dict
        struct_dict["fields"] = new_fields

    for _class_name, class_dict in api["classes"].items():
        # Class property names
        new_properties = {}
        for prop_name, prop_dict in class_dict["properties"].items():
            new_properties[to_snake_case(prop_name)] = prop_dict
        class_dict["properties"] = new_properties

        # Class method names
        new_methods = {}
        for method_name, method_dict in class_dict["methods"].items():
            new_method_name = to_snake_case(method_name)
            if new_method_name == "constructor":
                new_method_name = "__init__"
            new_methods[new_method_name] = method_dict

            # Class method arg names
            new_args = {}
            for arg_name, arg_dict in method_dict["args"].items():
                new_args[to_snake_case(arg_name)] = arg_dict
            method_dict["args"] = new_args
        class_dict["methods"] = new_methods


def to_snake_case(name, separator="_"):
    """Convert a name from camelCase to snake_case. Names that already are snake_case remain the same."""
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


def resolve_types(api, parser):
    """Resolve JS types to Python equivalents."""
    resolver = TypeResolver(api, parser.typedefs)
    resolve_type = resolver.resolve_type

    # Struct fields
    for _struct_name, struct_dict in api["structs"].items():
        for _field_name, field_dict in struct_dict["fields"].items():
            field_dict["type"] = resolve_type(field_dict["type"])

    for _class_name, class_dict in api["classes"].items():
        # Class properties
        for _prop_name, prop_dict in class_dict["properties"].items():
            prop_dict["type"] = resolve_type(prop_dict["type"])

        # Class method args and return
        for _method_name, method_dict in class_dict["methods"].items():
            method_dict["returns"]["type"] = resolve_type(
                method_dict["returns"]["type"]
            )
            for _arg_name, arg_dict in method_dict["args"].items():
                arg_dict["type"] = resolve_type(arg_dict["type"])


class TypeResolver:
    typemap = {
        "DOMString": "str",
        "USVString": "str",
        "long": "int",
        "unsigned long": "int",
        "unsigned long long": "int",
        "[Clamp] unsigned short": "int",
        "double": "float",
        "boolean": "bool",
        "ImageBitmap": "ArrayLike",
        "ImageData": "ArrayLike",
        "VideoFrame": "ArrayLike",
        "AllowSharedBufferSource": "ArrayLike",
        "[AllowShared] Uint32Array": "ArrayLike",
        "GPUPipelineConstantValue": "float",
        "GPUExternalTexture": "object",
        "undefined": "None",
        "ArrayBuffer": "ArrayLike",
        "WGSLLanguageFeatures": "set",
        "GPUSupportedFeatures": "set",
        "GPUSupportedLimits": "dict",
        "EventHandler": "None",
        "HTMLCanvasElement": "CanvasLike",
        "OffscreenCanvas": "CanvasLike",
    }

    def __init__(self, apidict: dict, typedefs: dict):
        self.typedefs = typedefs
        self.apidict = apidict

    def resolve_type(self, typename: str) -> str:
        """Resolve a type to a suitable name that is also valid so that the linter
        won't complain when this is used as a type annotation.
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

        name = self.typemap.get(name, name)

        # Is this a case for which we need to recurse?
        if (
            name.startswith("sequence<") or name.startswith("FrozenArray<")
        ) and name.endswith(">"):
            name = name.split("<")[-1].rstrip(">")
            name = self.resolve_type(name).strip("'")
            return f"list[{name}]"
        elif name.startswith("record<") and name.endswith(">"):
            name = name.split("<")[-1].rstrip(">")
            names = [self.resolve_type(t).strip("'") for t in name.split(",")]
            return f"dict[{', '.join(names)}]"
        elif " or " in name:
            # Clean
            name = name.strip("()")
            names = [self.resolve_type(t).strip("'") for t in name.split(" or ")]
            names = set(names)  # de-dupe
            has_none = "None" in names
            names.discard("None")
            names = sorted(names)
            if has_none:
                names.append("None")
            # Triage
            if len(names) == 1:
                return names[0]
            if (
                len(names) == 2
                and names[0] in ("list[int]", "list[float]")
                and names[1].endswith("Struct")
                and names[1].startswith(
                    ("structs.Origin", "structs.Extent", "structs.Color")
                )
            ):
                if names[1].endswith("ColorStruct"):
                    names[0] = "tuple[float, float, float, float]"
                elif names[1].endswith("2DStruct"):
                    names[0] = "tuple[int, int]"
                elif names[1].endswith("3DStruct"):
                    names[0] = "tuple[int, int, int]"
            return " | ".join(names)
        if name.startswith("Promise<") and name.endswith(">"):
            subname = name.split("<")[-1].rstrip(">")
            subname = self.resolve_type(subname)
            return f"GPUPromise[{subname}]"

        # Triage
        if name in builtins:
            return name  # ok
        elif name in ["ArrayLike", "CanvasLike"]:
            return name  # virtual types
        elif name in self.apidict["classes"]:
            return name
        elif name.startswith("HTML"):
            return "object"  # anything, we ignore this stuff anyway
        elif name in ["PredefinedColorSpace"]:
            return "str"
        else:
            assert name.startswith("GPU"), f"Unknown type: {name}"
            name = name[3:]
            name = name[:-4] if name.endswith("Dict") else name
            if name in self.apidict["flags"]:
                return f"flags.{name}Flags"
            elif name in self.apidict["enums"]:
                return f"enums.{name}Enum"
            elif name in self.apidict["structs"]:
                return f"structs.{name}Struct"
            else:
                # When this happens, update the code above or the typemap
                raise RuntimeError("Encountered unknown IDL type: ", name)


# ------------------------------ The parser code ------------------------------


class Attribute:
    """A little object to hold a function argument or struct field."""

    def __init__(self, line: str, source: Literal["struct", "param"]):
        self.line = line.strip().strip(",;").strip()
        self.source = source

        # Detect default value, store as a str that represents a Python literal.
        # Default empty string means there is no default; required arg/field.
        # Note that if default *represents* an empty string, it would be '""'.
        default = ""
        arg = self.line
        if "=" in arg:
            arg, default = arg.rsplit("=", 1)
            arg, default = arg.strip(), default.strip()
        if default:
            if default in ["false", "true"]:
                default = default.capitalize()
            if default == "null":
                default = "None"
        arg_type, arg_name = arg.strip().rsplit(" ", 1)

        # Handling required vs optional differs between structs fields and function params.
        # Struct fields are optional unless marked as required, see https://webidl.spec.whatwg.org/#required-dictionary-member
        # Function params are required unless marked as optional, see https://webidl.spec.whatwg.org/#dfn-optional-argument
        if source == "struct":
            if arg_type.startswith("required "):
                required = True
                arg_type = arg_type.split(" ", 1)[1]
                # required args should not have a default
                assert not default
            else:
                required = False
                default = default or "None"
        elif source == "param":
            if arg_type.startswith("optional "):
                required = False
                arg_type = arg_type.split(" ", 1)[1]
                default = default or "None"
            else:
                # but otherwise they are required
                required = True
                assert not default
        else:
            raise RuntimeError(f"Invalid Attribute source: {source!r}")

        assert isinstance(default, str)
        assert bool(default) == (not required)

        self.name = arg_name
        self.idl_type = arg_type
        self.default = default

    def __repr__(self):
        return f"<Attribute from {self.source} '{self.idl_type} {self.name}'>"

    def to_dict(self) -> dict:
        return {"type": self.idl_type, "default": self.default}


class Interface:
    """A class definition, or a flag/bitmask during parsing."""

    def __init__(self, name, bases):
        self.bases = bases
        self.constants = {}
        self.properties = {}  # name -> type
        self.methods = {}


class Function:
    """A function definition."""

    def __init__(self, name, line):
        self.name = name
        self.line = line
        self.args = None
        self.return_type = None


class IdlParser:
    """An object that can be used to walk over a str in an easy way.

    This parser has the following attributes:

    * flags: a dict mapping the (neutral) flag name to a dict of field-value pairs.
    * enums: a dict mapping the (Pythonic) enum name to a dict of field-value pairs.
    * structs: a dict mapping the (Pythonic) struct name to a dict of Attribute
      objects.
    * classes: a dict mapping the (normalized) class name an Interface object.

    """

    def __init__(self, source):
        self.source = self._pre_process(source)
        self._length = len(self.source)
        self._pos = 0

    def _reset(self):
        self._pos = 0

    def _end_reached(self):
        return self._pos >= self._length

    def _read_until(self, char):
        start = self._pos
        while self._pos < self._length:
            c = self.source[self._pos]
            self._pos += 1
            if c == char:
                return self.source[start : self._pos]
        return ""

    def _read_line(self):
        return self._read_until("\n")

    def _peek_line(self):
        char = "\n"
        start = pos = self._pos
        while pos < self._length:
            c = self.source[pos]
            pos += 1
            if c == char:
                return self.source[start:pos]
        return ""

    def parse(self, verbose=True):
        """Parse the IDL file and build up an internal structure."""
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
            f_count = sum(len(cls.methods) for cls in self.classes.values())
            print(
                f"The webgpu.idl defines {len(self.classes)} classes with {f_count} methods"
            )
            keys = "flags", "enums", "structs"
            stats = ", ".join(f"{len(getattr(self, key))} {key}" for key in keys)
            print("The webgpu.idl defines " + stats)

    def _pre_process(self, text):
        """Pre-process the text to make it a bit easier to parse.
        Be careful to keep line numbers the same
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
        while not self._end_reached():
            line = self._read_line()

            if not line.strip():
                pass
            elif line.startswith("typedef "):
                # Get the important bit
                value = line.split(" ", 1)[-1]
                if value.startswith("["):
                    value = value.split("]")[-1]
                # Parse
                if value.startswith("("):  # Union type
                    while ")" not in value:
                        value = value.rstrip() + " " + self._read_line().lstrip()
                    assert value.count("(") == 1 and value.count(")") == 1
                    value = value.split("(")[1]
                    val, _, key = value.partition(")")
                else:  # Singleton type
                    val, _, key = value.rpartition(" ")
                key = key.strip().strip(";").strip()
                self.typedefs[key] = val.strip()
            elif line.startswith(("namespace ", "interface ", "partial interface ")):
                # A class or a set of flags, we don't know yet, and resolve it during post-processing
                # Collect lines that define this interface
                while "{" not in line:
                    line = line.rstrip() + " " + self._read_line().lstrip()
                lines = [line]
                while not line.startswith("};"):
                    line = self._read_line()
                    lines.append(line)
                classname_raw, _, base_raw = lines[0].split("{")[0].partition(":")
                classname = classname_raw.split()[-1]
                # Collect base classes
                based_on = list(base_raw.split())
                while self._peek_line().startswith(classname + " includes "):
                    line = self._read_line()
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
                        continue
                    elif line.startswith("const "):
                        parts = line.strip(";").split()
                        assert len(parts) == 5
                        assert parts[-2] == "="
                        name = parts[2]
                        val = int(parts[-1], 16)
                        interface.constants[name] = val
                    elif "attribute " in line:
                        subline = line.partition("attribute")[2].strip().strip(";")
                        name = subline.split()[-1]
                        type = subline.split(name)[0].strip()
                        interface.properties[name] = type
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
                        interface.methods[funcname] = Function(funcname, line)
            elif " includes " in line:
                # Definition of class inheretance
                parts = line.strip(";").split()
                assert len(parts) == 3 and parts[1] == "includes"
                classname, _, base = parts
                if classname not in self._interfaces:
                    self._interfaces[classname] = Interface(classname, [])
                self._interfaces[classname].bases.append(parts[2])
            elif line.startswith("enum "):
                # Enums
                line += self._read_until("}") + self._read_line()
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
                # Structs / descriptors
                while "{" not in line:
                    line = line.rstrip() + self._read_line()
                assert line.count("{") == 1 and line.count("}") == 0
                lines = [line]
                while not line.startswith("};"):
                    line = self._read_line()
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

                    attribute = Attribute(line, "struct")
                    d[attribute.name] = attribute
                self.structs[name] = d
            elif line.startswith(("[Exposed=", "[Serializable]")):
                pass
            else:
                raise RuntimeError("Unknown line:", line.rstrip())

    def _post_process(self):
        """We don't do any name format normalization in the parser code itself;
        we do that here.
        """

        # Divide flags and class definitions
        for name, interface in self._interfaces.items():
            if interface.constants:
                self.flags[name] = interface.constants
            elif name in ("Navigator", "WorkerNavigator"):
                pass
            else:
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

        # Resolve method signatures
        for interface in self.classes.values():
            for funcname in list(interface.methods.keys()):
                func = interface.methods[funcname]
                args, return_type = self._resolve_signature(func.line)
                func.return_type = return_type or "None"
                func.args = args

    def _resolve_signature(self, idl_line: str):
        # Get arg names and types
        args = idl_line.split("(", 1)[1].split(")", 1)[0].split(",")
        args = [Attribute(arg, "param") for arg in args if arg.strip()]
        return_type = idl_line.split()[0]
        if return_type.startswith("[NewObject]"):
            # [NewObject] can be skipped: https://webidl.spec.whatwg.org/#NewObject
            return_type = idl_line.split()[1]
        if return_type.startswith("constructor"):
            # this means __init__ which essentially returns None (used with Error Messages)
            return_type = None

        return args, return_type


if __name__ == "__main__":
    extract_idl()
