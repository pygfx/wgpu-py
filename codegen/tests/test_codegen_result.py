""" Test some aspects of the generated code.
"""

from codegen.files import read_file


def test_async_methods_and_props():
    # Test that only and all aync methods are suffixed with '_async'

    for fname in ["_classes.py", "backends/wgpu_native/_api.py"]:
        code = read_file(fname)
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("def "):
                assert "async" not in line, line
            elif line.startswith("async def "):
                name = line.split("def", 1)[1].split("(")[0].strip()
                assert name.endswith("_async"), line
