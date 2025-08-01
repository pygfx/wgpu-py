from cffi import FFI

from codegen.utils import print, remove_c_comments
from codegen.files import read_file


_parser = None


def _get_wgpu_header():
    """Func written so we can use this in both wgpu_native/_ffi.py and codegen/hparser.py"""
    # Read files
    lines1 = []
    lines1.extend(read_file("resources", "webgpu.h").splitlines())
    lines1.extend(read_file("resources", "wgpu.h").splitlines())
    # Deal with pre-processor commands, because cffi cannot handle them.
    # Just removing them, plus a few extra lines, seems to do the trick.
    lines2 = []
    for line in lines1:
        if line.startswith("#define ") and len(line.split()) > 2 and "0x" in line:
            line = line.replace("(", "").replace(")", "")
        elif line.startswith("#"):
            continue
        elif 'extern "C"' in line:
            continue
        for define_to_drop in [
            "WGPU_EXPORT ",
            "WGPU_NULLABLE ",
            " WGPU_OBJECT_ATTRIBUTE",
            " WGPU_ENUM_ATTRIBUTE",
            " WGPU_FUNCTION_ATTRIBUTE",
            " WGPU_STRUCTURE_ATTRIBUTE",
        ]:
            line = line.replace(define_to_drop, "")
        lines2.append(line)
    return "\n".join(lines2)


def get_h_parser(*, allow_cache=True):
    """Get the global HParser object."""

    # Singleton pattern
    global _parser
    if _parser and allow_cache:
        return _parser

    source = _get_wgpu_header()

    # Create parser
    hp = HParser(source)
    hp.parse()
    _parser = hp
    return hp


class HParser:
    """Object to parse the webgpu.h/wgpu.h header files, by letting cffi do the heavy lifting."""

    def __init__(self, source):
        self.source = source

    def parse(self, verbose=True):
        self.flags = {}
        self.enums = {}
        self.structs = {}
        self.functions = {}

        self._parse_from_h()
        self._parse_from_cffi()

        if verbose:
            print(f"webgpu.h/wgpu.h define {len(self.functions)} functions")
            keys = "flags", "enums", "structs"
            stats = ", ".join(f"{len(getattr(self, key))} {key}" for key in keys)
            print("webgpu.h/wgpu.h define " + stats)

    def _parse_from_h(self):
        code = self.source

        # Collect enums and flags. This is easy.
        # Note that flags are first defined as enums and then redefined as flags later.
        i1 = i2 = i3 = i4 = 0
        while True:
            # Find enum
            i1 = code.find("typedef enum", i4)
            i2 = code.find("{", i1)
            i3 = code.find("}", i2)
            i4 = code.find(";", i3)
            if i1 < 0:
                break
            # Decompose "typedef enum XX {...} XX;"
            name1 = code[i1 + 13 : i2].strip()
            name2 = code[i3 + 1 : i4].strip()
            assert name1 == name2
            assert name1.startswith("WGPU")
            name = name1[4:]
            self.enums[name] = enum = {}
            code_block = code[i2 + 1 : i3].strip().strip(";")
            block = remove_c_comments(code_block).strip()
            for f in block.split(","):
                if not f:
                    continue  # no item after last comma
                key, _, val = f.partition("=")
                # Handle key
                key = key.strip()
                assert key.startswith("WGPU") and "_" in key, f"key={key}"
                key = key.split("_", 1)[1]
                # Turn value into an int
                val = val.strip()
                if val.startswith("0x"):
                    enum[key] = int(val, 16)
                elif "<<" in val:
                    val1, _, val2 = val.partition("<<")
                    enum[key] = int(val1) << int(val2)
                elif "|" in val:  # field is an OR of the earlier fields :/
                    keys = [k.strip().split("_", 1)[1] for k in val.split("|")]
                    val = 0
                    for k in keys:
                        val |= enum[k]
                    enum[key] = val
                else:
                    enum[key] = int(val)

        for line in code.splitlines():
            # collect flags
            # schme: typedef WGPUFlags WGPUFlagName;
            if line.startswith("typedef WGPUFlags "):
                parts = line.strip().strip(";").split()
                assert len(parts) == 3
                name = parts[-1]
                assert name.startswith("WGPU")
                name = name[4:]
                assert not name.endswith("Flags"), "XxxxFlags should not longer exist"
                assert name not in self.enums, "flags used to look like enums"
                self.flags[name] = {}

            # fill flags
            # schema: static const WGPUFlagName WGPUFlagName_Value = 0x0000000000000001;
            if line.startswith("static const"):
                line = remove_c_comments(line).strip()
                flag_name = line.removeprefix("static const").lstrip().split()[0]
                flag_key, _, val = (
                    line.removeprefix(f"static const {flag_name}")
                    .strip()
                    .rstrip(";")
                    .partition("=")
                )
                # Check / normalize flag_name
                assert flag_name.startswith("WGPU")
                flag_name = flag_name[4:]
                assert flag_name in self.flags
                # Check / normalize flag_key
                assert flag_key.startswith(f"WGPU{flag_name}_")
                flag_key = flag_key.partition("_")[2].strip()
                self.flags[flag_name][flag_key] = self._parse_val_to_int(val)

        # Collect structs. This is relatively easy, since we only need the C code.
        # But we don't deal with union structs.
        i1 = i2 = i3 = i4 = 0
        while True:
            # Find struct
            i1 = code.find("typedef struct", i4)
            i2 = code.find("{", i1)
            i3 = code.find("}", i2)
            i4 = code.find(";", i3)
            if i1 < 0:
                break
            # Only do simple structs, not Unions
            if 0 < code.find("{", i2 + 1) < i3:
                continue
            # Decompose
            name = code[i3 + 1 : i4].strip()
            self.structs[name] = struct = {}
            for f in code[i2 + 1 : i3].strip().strip(";").split(";"):
                f = remove_c_comments(f).strip()
                if not f:
                    continue  # probably last item ended with a comma
                parts = f.strip().split()
                typename = " ".join(parts[:-1])
                typename = typename.replace("const ", "")
                key = parts[-1].strip("*")
                struct[key] = typename

        # Collect functions. This is not too hard, since we only need the C code.
        i1 = i2 = i3 = 0
        while True:
            # Find function
            i1 = code.find("wgpu", i3)
            i2 = code.find("(", i1)
            i3 = code.find(");", i2)
            if i1 < 0:
                break
            # Extract name, and check whether we found something real
            name = code[i1:i2]
            if not (name and name.isidentifier()):
                i3 = i1 + 5
                continue
            # Decompose further
            i1 = code.rfind("\n", 0, i1)
            line = code[i1 : i3 + 2]
            line = " ".join(line.split())  # effective way to put on one line
            self.functions[name] = line

    def _parse_from_cffi(self):
        self.ffi = ffi = FFI()
        ffi.cdef(self.source)

        # Collect structs. We iterate over all types. Some will resolve
        # to C types, the rest are structs. The types for the struct
        # fields are reduced to the C primitives, making it less useful
        # for annotations. We update the structs that we've found by
        # parsing wgpu.h directly.
        for names in ffi.list_types():
            for name in names:
                # name = ffi.getctype(name) - no, keep original
                if name.startswith("WGPU") and not name.endswith("Impl"):
                    t = ffi.typeof(name)
                    if not hasattr(t, "fields"):
                        continue  # probably an enum
                    elif not t.fields:
                        continue  # base struct / alias
                    s = ffi.new(f"{name} *")
                    # Construct struct
                    struct = {}
                    for key, field in t.fields:
                        typename = field.type.cname
                        # typename = ffi.getctype(typename)
                        if typename.startswith("WGPU"):
                            val = typename  # Enum or struct
                        else:
                            val = type(getattr(s, key)).__name__
                        struct[key] = val
                    # Update
                    if name not in self.structs:
                        self.structs[name] = struct
                    else:
                        ori_struct = self.structs[name]
                        assert set(struct) == set(ori_struct)
                        for key, val in struct.items():
                            if ori_struct[key] != val:
                                if val.startswith("_"):  # _CDataBase
                                    pass
                                elif ori_struct[key].startswith("WGPU"):
                                    if "/" not in ori_struct[key]:
                                        ori_struct[key] += "/" + val
                                else:
                                    ori_struct[key] = val
                    # Make copies
                    alt_name = name
                    while alt_name != ffi.getctype(alt_name):
                        alt_name = ffi.getctype(alt_name)
                        self.structs[alt_name] = self.structs[name]

    def _parse_val_to_int(self, val):
        """
        little helper to parse complex values for enums.
        """
        val = val.strip(" ()")
        if "|" in val:
            # recursively handle the "OR" values?
            res_val = 0
            for sub_val in val.split("|"):
                res_val |= self._parse_val_to_int(sub_val)
            return res_val
        if val.startswith("0x"):
            return int(val, 16)
        elif "<<" in val:
            val1, _, val2 = val.partition("<<")
            return int(val1) << int(val2)
        else:
            return int(val)
