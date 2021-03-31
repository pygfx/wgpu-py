import os

from cffi import FFI
from cffi.model import EnumType

from codegen.utils import lib_dir


class HParser:
    """Object to parse the wgpu.h header file."""

    def __init__(self):
        pass

    def _get_wgpu_h(self):
        lines = []
        with open(os.path.join(lib_dir, "resources", "wgpu.h")) as f:
            for line in f.readlines():
                if not line.startswith(
                    (
                        "#include ",
                        "#define WGPU_LOCAL",
                        "#define WGPUColor",
                        "#define WGPUOrigin3d_ZERO",
                        "#if defined",
                        "#endif",
                    )
                ):
                    lines.append(line)
        return "".join(lines)

    def parse(self):

        self.flags = flags = {}
        self.enums = enums = {}

        self.ffi = ffi = FFI()
        ffi.cdef(self._get_wgpu_h())
        # lib = ffi.dlopen(os.path.join(lib_dir, "resources", "wgpu_native-release.dll"))

        # Warning: we access private ffi stuff here

        # Collect enums and functions
        for key, (tp, _) in ffi._parser._declarations.items():
            tag, name = key.split(" ", 1)
            if isinstance(tp, EnumType):
                enums[name[4:]] = fields = {}
                for enumname, val in zip(tp.enumerators, tp.enumvalues):
                    fields[enumname[len(name) + 1 :]] = val
            elif tag == "function":
                pass  # print("function", name)

        # Collect flags by iterationg over constants that are not enums
        for key, value in ffi._parser._int_constants.items():
            if key.startswith("WGPU"):
                if key.upper() == key or "_" not in key:
                    continue
                name, _, field = key.partition("_")
                name = name[4:]  # strip "WGPU"
                if name in enums:
                    continue
                fields = flags.setdefault(name, {})
                fields[field] = value


# for name in ffi.list_types():
#     print(name)
#     s = ffi.new(f"{name} *")
