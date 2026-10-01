"""Generate the cffi cdef for wgpu_dawn from a webgpu.h, and diff two headers.

The cdef is generated from Emdawnwebgpu's ``webgpu.h``. At the same Dawn tag
this header is a strict subset of Dawn's native ``webgpu.h`` (see ``diff``), so
the same cdef, and therefore the same compiled binding layer and the same Python
code, works for both targets. Because cffi runs in API (out-of-line) mode, the C
compiler checks the cdef against the real header of each target at build time
(struct layouts, enum values, function signatures); constants are declared as
``...`` so they take the value of the target (e.g. ``WGPU_STRLEN`` is
``SIZE_MAX``, which differs between wasm32 and 64-bit native).

Usage::

    python gen_cdef.py cdef path/to/emdawnwebgpu_pkg/webgpu/include/webgpu/webgpu.h > ../src/wgpu_dawn/webgpu_cdef.h
    python gen_cdef.py native-extra path/to/dawn/webgpu.h path/to/emdawn/webgpu.h > ../src/wgpu_dawn/webgpu_native_extra_cdef.h
    python gen_cdef.py diff path/to/emdawn/webgpu.h path/to/dawn/webgpu.h
"""

import re
import subprocess
import sys


def make_cdef(header, cc="cc"):
    src = open(header).read()
    consts = sorted(set(re.findall(r"^#define (WGPU_[A-Z0-9_]+) \((.*)\)$", src, re.M)))
    # Integer constants only (cffi's "..." defines are integers)
    consts = [c for c, v in consts if not c.endswith("_INIT") and "NAN" not in v]
    body = re.sub(r"^#include .*$", "", src, flags=re.M)
    # dawn/webgpu.h redirects to the Emscripten header when __EMSCRIPTEN__ is set
    body = body.replace("#ifdef __EMSCRIPTEN__", "#if 0")
    out = subprocess.check_output(
        [cc, "-E", "-P", "-x", "c", "-", "-DWGPU_SKIP_PROCS"], input=body.encode()
    ).decode()
    text = "\n".join(ln for ln in out.splitlines() if ln.strip())
    text = re.sub(r"__attribute__\(\(.*?\)\)\s*", "", text)
    lines = [f"#define {c} ..." for c in consts]
    lines.append(text)
    # One Python entry point per callback type, used by the backend to receive
    # all async results (cffi "extern Python": no runtime code generation, which
    # is what makes callbacks work under WebAssembly).
    for name, args in re.findall(
        r"^typedef void \(\*(WGPU\w+Callback)\)\((.*?)\)\s*;", text, re.M
    ):
        lines.append(f'extern "Python" void _wgpu_dawn_{name}({args});')
    return "\n".join(lines) + "\n"


# Native-only declarations that wgpu.backends.dawn uses (window-system surfaces).
NATIVE_EXTRA_STRUCTS = [
    "WGPUSurfaceSourceXlibWindow",
    "WGPUSurfaceSourceWaylandSurface",
    "WGPUSurfaceSourceXCBWindow",
    "WGPUSurfaceSourceMetalLayer",
    "WGPUSurfaceSourceWindowsHWND",
]


def make_native_extra_cdef(native_header, shared_header, cc="cc"):
    """Extra cdef for native builds only (not available in Emdawnwebgpu)."""
    text = make_cdef(native_header, cc)
    shared = make_cdef(shared_header, cc)
    lines = []
    for name in NATIVE_EXTRA_STRUCTS:
        m = re.search(
            r"^typedef struct %s \{.*?^\} %s\s*;" % (name, name), text, re.M | re.S
        )
        lines.append(m.group(0))
        # The sType must be a member of WGPUSType, which in the shared cdef
        # only has the members of the Emdawnwebgpu header.
        stype = f"WGPUSType_{name[len('WGPU') :]}"
        if not re.search(r"\b%s\b" % stype, shared):
            lines.append(f"static const int {stype};")
    return "\n".join(lines) + "\n"


def declarations(cdef):
    from cffi import FFI

    ffi = FFI()
    ffi.cdef(cdef)
    return ffi._parser._declarations


def describe(tp):
    if not hasattr(tp, "_get_c_name"):
        return repr(tp)
    name = type(tp).__name__
    if name in ("StructType", "UnionType"):
        return [
            (n, t._get_c_name())
            for n, t in zip(tp.fldnames or (), tp.fldtypes or (), strict=True)
        ]
    if name == "EnumType":
        return list(zip(tp.enumerators, tp.enumvalues, strict=True))
    return tp._get_c_name()


def diff(header_a, header_b):
    a = declarations(make_cdef(header_a))
    b = declarations(make_cdef(header_b))
    only_a = sorted(k for k in a if k not in b)
    only_b = sorted(k for k in b if k not in a)
    print(f"declarations: A={len(a)} B={len(b)} common={len(set(a) & set(b))}")
    print(f"only in A ({len(only_a)}):", " ".join(only_a))
    print(f"only in B ({len(only_b)})")
    for k in sorted(set(a) & set(b)):
        da, db = describe(a[k][0]), describe(b[k][0])
        if da == db:
            continue
        if isinstance(da, list) and isinstance(db, list):
            removed, added = set(da) - set(db), set(db) - set(da)
            kind = "B adds members" if not removed else "INCOMPATIBLE"
            print(f"  {k}: {kind}; A-B={sorted(removed)} B-A: {len(added)} items")
        else:
            print(f"  {k}: INCOMPATIBLE\n    A: {da}\n    B: {db}")


if __name__ == "__main__":
    if sys.argv[1] == "cdef":
        sys.stdout.write(make_cdef(sys.argv[2]))
    elif sys.argv[1] == "native-extra":
        sys.stdout.write(make_native_extra_cdef(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
