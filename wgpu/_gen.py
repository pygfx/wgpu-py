"""
Script to parse wgpu.h and generate wgpu.py

Source: https://github.com/gfx-rs/wgpu/blob/master/ffi/wgpu.h
"""

import os


class TextOb:
    """ An object that can be used to read from a str in a nice way.
    """

    def __init__(self, text):
        self._text = text
        self._length = len(text)
        self._pos = 0

    def end_reached(self):
        return self._pos >= self._length

    def read_until(self, char):
        start = self._pos
        while self._pos < self._length:
            c = self._text[self._pos]
            self._pos += 1
            if c == char:
                return self._text[start:self._pos]
        return ""

    def readline(self):
        return self.read_until("\n")


# %% Parse

hfile = TextOb(open("wgpu.h", "rb").read().decode())

constants = {}
enums = {}
structs = {}
functions = {}
types = {}

unknown_lines = []


while not hfile.end_reached():

    line = hfile.readline()

    if not line.strip():
        pass
    elif line.startswith("/*"):
        if "*/" in line:
            pass
        else:
            raise RuntimeError("Cannot handle multiline comments yet.")
    elif line.startswith("#include "):
        pass
    elif line.startswith("#if !defined(WGPU_REMOTE)") or line.startswith("#if defined(WGPU_LOCAL)"):
        pass
    elif line.startswith("#endif"):
        pass
    elif line.startswith("#define "):
        parts = line.split()
        if len(parts) == 3:
            constants[parts[1].strip()] = int(parts[2].strip())
        elif "WGPU_LOCAL" in line:
            pass
        else:
            unknown_lines.append(line)
    elif line.startswith("typedef enum {"):
        line += hfile.read_until("}") + hfile.readline()
        lines = line.strip().split("\n")
        name = lines[-1].split("}", 1)[1].strip("; ")
        d = {}
        for i, line in enumerate(lines[1:-1]):
            key, _, val = line.strip().strip(",;").partition("=")
            val = val.strip()
            if not val:
                val = i
            d[key.strip()] = int(val)
        enums[name] = d
    elif line.startswith("typedef struct"):
        line += hfile.read_until("}") + hfile.readline()
        lines = line.strip().split("\n")
        name = lines[-1].split("}", 1)[1].strip("; ")
        if not name:
            line += hfile.read_until("}") + hfile.readline()
            unknown_lines.append(line)
            continue
        d = {}
        for line in lines[1:-1]:
            arg = line.strip().strip(",;")
            if arg.startswith("const "):
                arg = arg[6:]
            arg_type, arg_name = arg.strip().split()
            d[arg_name.strip(' *')] = arg_type
        structs[name] = d
    elif line.startswith("typedef "):
        parts = line.strip().strip(";").split()
        if len(parts) == 3:
            types[parts[2]] = parts[1]
        else:
            unknown_lines.append(line)
    elif (line.startswith("void ") or line.startswith("WGPU")) and "wgpu_" in line:
        if ")" not in line:
            line += hfile.read_until(")") + hfile.readline()
        # Get name
        name = line.split("(")[0].strip().split()[-1].strip()
        # Return type
        rtype = line.strip().split()[0]
        # Parse args
        assert line.count("(") == 1
        raw_args = line.split("(")[1].split(")")[0].split(",")
        args = []
        for arg in raw_args:
            arg = arg.strip()
            if arg.startswith("const "):
                arg = arg[6:]
            arg_type, arg_name = arg.strip().split()
            args.append((arg_name.strip(' *'), arg_type))
        # Collect
        functions[name] = dict(line=line, args=args, rtype=rtype)
    else:
        unknown_lines.append(line)


def pythonise_type(t):
    t = types.get(t, t)
    t = types.get(t, t)  # because can be XX -> XXDummy -> uint32_t
    if t in ("float", "double"):
        return "float"
    elif t in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
        return "int"
    elif t.endswith("_t"):
        return t[:-2]
    elif t.startswith("WGPU"):
        return t[4:]
    else:
        return t

def type_annotation(t):
    t = pythonise_type(t)
    if t in ("int", "float"):
        return f": {t}"
    elif t == "void":
        return ""
    else:
        return f": {t!r}"

def type_to_ctype(t):
    t = types.get(t, t)
    t = types.get(t, t)
    t = types.get(t, t)  # because can be XX -> XXDummy -> uint32_t
    if t == "void":
        return "ctypes.c_void_p"
    elif t in ("bool", "float", "double"):
        return "ctypes.c_" + t
    elif t in ("uint8_t", "int32_t", "int64_t", "uint32_t", "uint64_t"):
        return "ctypes.c_" + t[:-2]
    elif t in ("uintptr_t", ):
        return "ctypes.POINTER(ctypes.c_uint64)"  # todo: probably
    elif t == "WGPURawString":
        return "ctypes.c_char_p"
    elif t in ("WGPUBufferMapReadCallback", "WGPUBufferMapWriteCallback"):
        return "ctypes.c_void_p"  # todo: function pointer
    elif t in structs:
        return t
    elif t in enums or (t + "_Tag") in enums:
        return "ctypes.c_int64"  # todo: --->>>> uint32 causes access violation, ??? but with cffi it seems enums are 4 bytes ...
    else:
        raise NotImplementedError()

# Summarize
if unknown_lines:
    print(f"===== Could not parse {len(unknown_lines)} lines:")
    for line in unknown_lines:
        print(line.rstrip())
    print("=====")
else:
    print(f"All lines where parsed")
print(f"Found {len(constants)} constants")
print(f"Found {len(enums)} enums")
print(f"Found {len(structs)} structs")
print(f"Found {len(functions)} functions")


# %% Generate abstract API

module_doc = """
THIS CODE IS AUTOGENERATED - DO NOT EDIT
"""

class_doc = """
Abstract base class for the WebGPU API.
"""


pylines = []

pylines.append(f'"""\n{module_doc.strip()}\n"""\n')
pylines.append("\nclass BaseWGPU:")
pylines.append(f'    """ {class_doc.strip()}\n    """')

pylines.append(f"\n    # %% Functions ({len(functions)})")
for name, d in functions.items():
    assert name.startswith("wgpu_")
    name = name[5:]
    args = [arg[0] + type_annotation(arg[1]) for arg in d['args']]
    pylines.append(f"\n    def {name}(self, {', '.join(args)}):")
    pylines.append(f'        """')
    for line in d['line'].strip().splitlines():
        pylines.append("        " + line)
    pylines.append(f'        """')
    pylines.append("        raise NotImplementedError()")

pylines.append(f"\n    # %% Structs ({len(structs)})\n")
for name, vals in structs.items():
    assert name.startswith("WGPU")
    name = name[4:]
    c_args = [t + " " + key for key, t in vals.items()]
    py_args = [key + type_annotation(t) for key, t in vals.items()]
    dict_args = [f'"{key}": {key}' for key in vals.keys()]
    pylines.append(f"\n    def create_{name}(self, *, {', '.join(py_args)}):")
    pylines.append(f'        """ ' + ", ".join(c_args) + ' """')
    pylines.append("        return {"+ ", ".join(dict_args) + "}")

pylines.append(f"\n    # %% Constants ({len(constants)})\n")
for name, val in constants.items():
    assert name.startswith("WGPU")
    name = name[4:]
    pylines.append(f"    {name} = {val!r}")

pylines.append(f"\n    # %% Enums ({len(enums)})\n")
for name, subs in enums.items():
    assert name.startswith("WGPU")
    name = name[4:]
    pylines.append(f"    # {name}")
    for sub_name, sub_val in subs.items():
        assert sub_name.startswith("WGPU")
        sub_name = sub_name[4:]
        # assert sub_name.startswith(name)
        pylines.append(f"    {sub_name} = {sub_val!r}")

pylines.append("")

# Write

with open("wgpu.py", "wb") as f:
    f.write("\n".join(pylines).encode())
print("Written to wgpu.py")


# %% Generate ctypes wrapper

# todo: does not work (yet), ffi is probably simpler to get right

module_doc = """
THIS CODE IS AUTOGENERATED - DO NOT EDIT
"""

module_preamble = """
import os
import ctypes

from .wgpu import BaseWGPU


_lib_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wgpu_native-release.dll")
_lib = ctypes.windll.LoadLibrary(_lib_filename)


def dict_to_struct(d, struct_class):
    args = []
    for key, t in struct_class._fields_:
        val = d[key]
        if issubclass(t, ctypes.Structure):
            args.append(dict_to_struct(val, t))
        else:
            args.append(val)
    return struct_class(*args)


def struct_to_dict(s):
    d = {}
    for key, t in s._fields_:
        val = getattr(s, key)
        if isinstance(val, ctypes.Structure):
            val = struct_to_dict(val)
        d[key] = val
    return d
"""

class_doc = """
WebGPU API implemented using the C-API dll of wgpu-rs.
"""


pylines = []

pylines.append(f'"""\n{module_doc.strip()}\n"""\n')
pylines.append(module_preamble.rstrip() + "\n\n")

# Define structs
for name, vals in structs.items():
    assert name.startswith("WGPU")
    pylines.append(f"class {name}(ctypes.Structure):")
    pylines.append("    _fields_ = [")
    for key, t in vals.items():
        pylines.append(f'        ("{key}", {type_to_ctype(t)}),')
    pylines.append("    ]\n")
pylines.append("")

# Define function args types
for name, d in functions.items():
    start_len = len(pylines)
    assert name.startswith("wgpu_")
    argtypes = [type_to_ctype(arg[1]) for arg in d['args']]
    pylines.append(f"_lib.{name}.argtypes = (" + ", ".join(argtypes) + ", )")
    if d["rtype"] != "void":
        pylines.append(f"_lib.{name}.restype = " + type_to_ctype(d["rtype"]))
    if name.startswith("wgpu_create_surface_from_"):
        for i in range(start_len, len(pylines)):
            pylines[i] = "    " + pylines[i]
        pylines.insert(start_len, "try:  # OS-specific")
        pylines.append("except AttributeError:\n    pass")
pylines.append("")

pylines.append("\nclass RsWGPU(BaseWGPU):")
pylines.append(f'    """ {class_doc.strip()}\n    """')
for name, d in functions.items():
    assert name.startswith("wgpu_")
    pyname = name[5:]
    args = [arg[0] + type_annotation(arg[1]) for arg in d['args']]
    pylines.append(f"\n    def {pyname}(self, {', '.join(args)}):")
    pylines.append(f'        """')
    for line in d['line'].strip().splitlines():
        pylines.append("        " + line)
    pylines.append(f'        """')
    # call
    args = []
    for key, t in d['args']:
        if t in structs:
            args.append(f"dict_to_struct({key}, {t})")
        else:
            args.append(key)
    pylines.append(f'        return _lib.{name}(' + ", ".join(args) + ")")


pylines.append("")

# Write

with open("wgpu_ctypes.py", "wb") as f:
    f.write("\n".join(pylines).encode())
print("Written to wgpu_ctypes.py")


# %% Generate cffi wrapper

# todo: use ffi.emit_python_code so we dont have to parse the header file at runtime

module_doc = """
THIS CODE IS AUTOGENERATED - DO NOT EDIT
"""


module_preamble = """
import os
from os import path
from cffi import FFI

from .wgpu import BaseWGPU

os.environ["RUST_BACKTRACE"] = "0"

HERE = path.dirname(path.realpath(__file__))

ffi = FFI()

# read file
lines = []
with open(path.join(HERE, 'wgpu.h')) as f:
    for line in f.readlines():
        if not line.startswith(("#include ", "#define WGPU_LOCAL", "#define WGPUColor", "#define WGPUOrigin3d_ZERO", "#if defined", "#endif")):
            lines.append(line)


# configure cffi
ffi.cdef("".join(lines))
ffi.set_source("whatnameshouldiusehere", None)

_lib = ffi.dlopen(path.join(HERE, "wgpu_native-release.dll"))

def dict_to_struct(d, struct, refs):
    # return ffi.new(struct + " *", d)
    if d is None:
        return ffi.NULL
    is_flat = True
    for val in d.values():
        if isinstance(val, (tuple, list, dict, str)):
            is_flat = False
    if not is_flat:
        s = ffi.new(struct + " *")
        for key, sub_struct in _struct_info[struct]:
            val = d[key]
            if isinstance(val, (tuple, list)):
                assert sub_struct
                val2 = []
                if val and isinstance(val[0], dict):
                    for v in val:
                        val2.append(dict_to_struct(v, sub_struct, refs))
                    refs.extend(val2)
                    val = [v[0] for v in val2]
                    val = ffi.new(sub_struct + " []", val)
                else:
                    val = ffi.new(sub_struct + " []", val)
                refs.append(val)
            elif isinstance(val, dict):
                val = dict_to_struct(val, sub_struct, refs)
                refs.append(val)
                if "*" not in str(getattr(s, key)):
                    val = val[0]
            elif isinstance(val, str):
                val = ffi.new("char []", val.encode())
                refs.append(val)
            elif val is None:
                val = ffi.NULL
            setattr(s, key, val)
        return s
    else:
        return ffi.new(struct + " *", d)


"""

class_doc = """
WebGPU API implemented using the C-API dll of wgpu-rs, via cffi.
"""


pylines = []

pylines.append(f'"""\n{module_doc.strip()}\n"""\n')
pylines.append(module_preamble.rstrip() + "\n\n")

# Define structs
pylines.append("_struct_info = dict(")
for name, vals in structs.items():
    assert name.startswith("WGPU")
    line = f"    {name} = ["
    for key, t in vals.items():
        t = type_to_ctype(t)
        if t.startswith("ctypes.c_"):
            t = t[9:] + "_t"
        line += f"('{key}', '{t}'), "
    pylines.append(line + " ],")
pylines.append(")\n")

pylines.append("\nclass RsWGPU(BaseWGPU):")
pylines.append(f'    """ {class_doc.strip()}\n    """')
for name, d in functions.items():
    assert name.startswith("wgpu_")
    pyname = name[5:]
    args = [arg[0] + type_annotation(arg[1]) for arg in d['args']]
    pylines.append(f"\n    def {pyname}(self, {', '.join(args)}):")
    pylines.append(f'        """')
    for line in d['line'].strip().splitlines():
        pylines.append("        " + line)
    pylines.append(f'        """')
    # call
    pylines.append(f'        xx = []')
    args = []
    for key, t in d['args']:
        args.append(key)
        if t in structs:
            # args.append(f"ffi.new('{t} *', {key})")
            pylines.append(f"        {key} = dict_to_struct({key}, '{t}', xx)")
    pylines.append(f'        return _lib.{name}(' + ", ".join(args) + ")")


pylines.append("")

# Write

with open("wgpu_ffi.py", "wb") as f:
    f.write("\n".join(pylines).encode())
print("Written to wgpu_ffi.py")