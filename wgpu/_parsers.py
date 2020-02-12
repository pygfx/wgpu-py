"""
This module provides classes to parse the .idl and .h files defining
the WebGPU and wgpu API. These help us to generate code (definitions
of flags, enums and stucts) and to provide help for developers.

Naming conventions differ between IDL and header files. To deal with that, we
use a neutral name as key in some places, and convert to Python name in others:

* flags: name is CamelCase, field names are UPPERCASE, values are int.
  No changes required.
* enums: name is CamelCase, field names are snake_case, values are 'like-this'.
  We need to convert the field names of IDL (these are the same as the values).
* structs: their fields are the function arguments, so we never refer them by name.
* function args: name is snake_case. Needs conversion.
* function names: name is snake_case. Needs conversion. To uniquely identify
  functions (in .h the classname is part of the function name) the keys in the
  functions dict are neutral names.

"""


def to_neutral_name(name):
    """ Convert a name to the neutral name with no capitals, underscores or dots.
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
    """ Convert someName and some_name to the Python flavor.
    To convert function names and function argument names.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0 and name2[-1] != "_":
            name2 += "_"
        name2 += c2
    return name2


class BaseParser:
    """ An object that can be used to walk over a str in an easy way.

    Our parsers have the following attributes:

    * flags: a dict mapping the (neutral) flag name to a dict of field-value pairs.
    * enums: a dict mapping the (Pythonic) enum name to a dict of field-value pairs.
    * structs: a dict mapping the (Pythonic) struct name to a dict of StructField
      objects.
    * functions: a dict mapping the (normalized) func name to the line defining the
      function.

    """

    def __init__(self, text):
        self._text = text
        self._length = len(text)
        self._pos = 0

    def reset(self):
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

    def readline(self):
        return self.read_until("\n")

    def parse(self, verbose=False):
        self._pos = 0

        self.flags = {}
        self.enums = {}
        self.structs = {}
        self.functions = {}
        self.callbacks = {}
        self.types = {}

        self.unknown_lines = []

        if verbose:
            print(f"##### Parsing with {self.__class__.__name__} ...")

        self._parse()

        self._normalize()

        # Summarize
        if verbose:
            if self.unknown_lines:
                print(f"Could not parse {len(self.unknown_lines)} lines")
            else:
                print(f"All lines where parsed")
            print(f"Found {len(self.flags)} flags")
            print(f"Found {len(self.enums)} enums")
            print(f"Found {len(self.structs)} structs")
            print(f"Found {len(self.functions)} functions")
            print(f"Found {len(self.callbacks)} callbacks")

    def _parser(self):
        raise NotImplementedError()

    def _normalize(self):
        raise NotImplementedError()


class StructField:
    """ A little object to specify the field of a struct.
    """

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
        name = to_python_name(self.name)
        t = self.typename
        d = self.default
        if t not in ("bool", "int", "float", "str"):
            t = f"'{t}'"
        if d is not None:
            d = {"false": "False", "true": "True"}.get(d, d)
            return f"{name}: {t}={d}"
        else:
            return f"{name}: {t}"


# %% IDL


class IdlParser(BaseParser):
    """ Parse (part of) IDL files to obtain info about flags, enums and structs.
    """

    def _normalize(self):
        """ We don't do any name format normalization in the parser code itself;
        we do that here.
        """

        # Remove GPU prefix for flags, enums and structs
        for d in (self.flags, self.enums, self.structs):
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

        # Normalize function name to be a flat lowercase name without underscores
        for name in list(self.functions.keys()):
            assert name.startswith("GPU") and name.count(".") == 1
            self.functions[to_neutral_name(name)] = self.functions.pop(name)

    def _parse(self):

        # First resolve typedefs
        # mmm, actually, these GPU types also work as docs ...
        typedefs = {}
        # while not self.end_reached():
        #     line = self.readline()
        #     if line.startswith("typedef "):
        #         # Track typedefs to resolve types. Skip union types (with "or")
        #         line2 = line.split("]")[-1]
        #         if " or " in line:
        #             pass
        #         else:
        #             parts = line2.strip().strip(";").split()
        #             if len(parts) >= 2:
        #                 typedefs[parts[-1]] = " ".join(parts[:-1])
        # self.reset()

        while not self.end_reached():

            line = self.readline()

            if not line.strip():
                pass
            elif line.startswith("//"):
                pass
            elif line.startswith("/*"):
                if "*/" in line:
                    pass
                else:
                    raise RuntimeError("Cannot handle multiline comments yet.")
            elif line.startswith("interface "):
                lines = [line]
                while not line.startswith("};"):
                    line = self.readline()
                    lines.append(line)
                classname = lines[0].split("{")[0].split(":")[0].split()[-1]
                line_index = 0
                while line_index < len(lines) - 1:
                    line_index += 1
                    line = lines[line_index].strip()
                    if not line or line.startswith("//"):
                        continue
                    elif line.startswith("const ") and "Flags" in line:
                        parts = line.strip(";").split()
                        assert parts[-2] == "="
                        assert parts[1].endswith("Flags")
                        basename = parts[1][:-5]
                        name = parts[2]
                        val = int(parts[-1], 16)
                        self.flags.setdefault(basename, {})[name] = val
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
                        self.functions[classname + "." + funcname] = line
            elif line.startswith("enum "):
                line += self.read_until("}") + self.readline()
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
                    line = self.readline()
                    lines.append(line)
                name = lines[0].split(" ", 1)[1].strip("{ \t\r\n")
                # TODO: unused
                # if "GPUDeviceDescriptor" in name:
                #     a = 323
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
                    # TODO: unused
                    # is_required = False
                    default = None
                    if "=" in arg:
                        arg, default = arg.rsplit("=", 1)
                        arg, default = arg.strip(), default.strip()
                    arg_type, arg_name = arg.strip().rsplit(" ", 1)
                    if arg_type.startswith("required "):
                        # TODO: unused
                        # is_required = True
                        arg_type = arg_type[9:]
                    arg_type = typedefs.get(arg_type, arg_type)
                    if arg_type in ["double", "float"]:
                        t = "float"
                    elif arg_type in ["long", "unsigned long", "unsigned long long"]:
                        t = "int"
                    elif arg_type in ["boolean"]:
                        t = "bool"
                    elif arg_type in ["DOMString", "DOMString?"]:
                        t = "str"
                    elif arg_type.startswith("GPU"):
                        t = arg_type
                        # todo: can in some cases resolve this to int/float via typedefs
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
            else:
                self.unknown_lines.append(line)


# %% C-header


class HParser(BaseParser):
    """ Parse (part of) .h files to obtain info about flags, enums and structs.
    """

    # def pythonise_type(self, t):
    #     t = self.types.get(t, t)
    #     t = self.types.get(t, t)  # because can be XX -> XXDummy -> uint32_t
    #     if t in ("float", "double"):
    #         return "float"
    #     elif t in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
    #         return "int"
    #     elif t.endswith("_t"):
    #         return t[:-2]
    #     elif t.startswith("WGPU"):
    #         return t[4:]
    #     else:
    #         return t

    # def type_annotation(self, t):
    #     t = self.pythonise_type(t)
    #     if t in ("int", "float"):
    #         return f": {t}"
    #     elif t == "void":
    #         return ""
    #     else:
    #         return f": {t!r}"

    # def type_to_ctype(self, t):
    #     while self.types.get(t, t) is not t:
    #         t = self.types.get(t, t)
    #     if t == "void":
    #         return "ctypes.c_void_p"
    #     elif t in ("bool", "float", "double"):
    #         return "ctypes.c_" + t
    #     elif t in ("uint8_t", "int32_t", "int64_t", "uint32_t", "uint64_t"):
    #         return "ctypes.c_" + t[:-2]
    #     elif t in ("uintptr_t", ):
    #         return "ctypes.POINTER(ctypes.c_uint64)"  # todo: probably
    #     elif t == "WGPURawString":
    #         return "ctypes.c_char_p"
    #     elif t in ("WGPUBufferMapReadCallback", "WGPUBufferMapWriteCallback",
    #                "WGPURequestAdapterCallback"):
    #         return "ctypes.c_void_p"  # todo: function pointer
    #     elif t in self.structs:
    #         return t
    #     elif t in self.enums:
    #         # todo: --->>>> uint32 causes access violation, ??? but with cffi it seems
    #         #               enums are 4 bytes ...
    #         return "ctypes.c_int64"
    #     # elif t == "WGPUBindingResource":
    #         # return "dunno"
    #     else:
    #         raise NotImplementedError()

    def _normalize(self):
        """ We don't do any name format normalization in the parser code itself;
        we do that here.
        """

        # Remove WGPU prefix for flags, enums and structs
        for d in (self.flags, self.enums, self.structs):
            for name in list(d.keys()):
                assert name.startswith("WGPU")
                new_name = name[4:]
                d[new_name] = d.pop(name)

        # Normalize function name to be a flat lowercase name without underscores
        for name in list(self.functions.keys()):
            assert name.startswith("wgpu") and "." not in name
            self.functions[to_neutral_name(name)] = self.functions.pop(name)

    def _parse(self):

        while not self.end_reached():

            line = self.readline()

            if not line.strip():
                pass
            elif line.startswith("//"):
                pass
            elif line.startswith("/*"):
                while "*/" not in line:
                    line += self.readline()
            elif line.startswith("#include "):
                pass
            elif line.startswith("#if !defined(WGPU_REMOTE)") or line.startswith(
                "#if defined(WGPU_LOCAL)"
            ):
                pass
            elif line.startswith("#endif"):
                pass
            elif line.startswith("#define "):
                parts = line.split()
                if len(parts) == 3:
                    basename, _, name = parts[1].partition("_")
                    val = int(parts[2].strip())
                    self.flags.setdefault(basename, {})[name] = val
                elif "WGPU_LOCAL" in line:
                    pass
                else:
                    self.unknown_lines.append(line)
            elif line.startswith("typedef enum {"):
                line += self.read_until("}") + self.readline()
                lines = line.strip().split("\n")
                name = lines[-1].split("}", 1)[1].strip("; ")
                d = {}
                for i, line in enumerate(lines[1:-1]):
                    key, _, val = line.strip().strip(",;").partition("=")
                    val = val.strip()
                    if not val:
                        val = i
                    key = key[len(name) + 1 :]
                    d[key.strip()] = int(val)
                self.enums[name] = d
            elif line.startswith("typedef struct"):
                assert line.count("{") == 1 and line.count("}") == 0
                nesting_level = 1
                while nesting_level > 0:
                    more_line = self.read_until("}") + self.readline()
                    line += more_line
                    nesting_level += more_line.count("{") - more_line.count("}")
                lines = line.strip().split("\n")
                name = lines[-1].split("}", 1)[1].strip("; ")
                assert name
                d = {}
                union = False
                for line in lines[1:-1]:
                    line = line.strip()
                    if not union:
                        if line.startswith("union {"):
                            union = True
                            continue
                    else:  # in a union
                        if line == "};":
                            union = False
                            continue
                    assert line.endswith(";")
                    arg = line.strip(",;")
                    if arg.startswith("const "):
                        arg = arg[6:]
                    arg_type, arg_name = arg.strip().split()
                    arg_name = arg_name.strip(" *")
                    if union:
                        line += " (in union)"
                    d[arg_name] = StructField(line, arg_name, arg_type)
                self.structs[name] = d
            elif line.startswith("typedef void (*") and "Callback" in line:
                name = line.split("(*", 1)[1].split(")")[0].strip()
                self.callbacks[name] = line.strip()
            elif line.startswith("typedef "):
                parts = line.strip().strip(";").split()
                if len(parts) == 3:
                    self.types[parts[2]] = parts[1]
                else:
                    self.unknown_lines.append(line)
            elif (
                line.startswith("void ") or line.startswith("WGPU")
            ) and "wgpu_" in line:
                if ")" not in line:
                    line += self.read_until(")") + self.readline()
                name = line.split("(")[0].strip().split()[-1].strip().strip("*")
                self.functions[name] = line
            else:
                self.unknown_lines.append(line)


if __name__ == "__main__":
    idl_parser = IdlParser(open("./resources/webgpu.idl", "rb").read().decode())
    idl_parser.parse()

    h_parser = HParser(open("./resources/wgpu.h", "rb").read().decode())
    h_parser.parse()
