"""Test some aspects of the generated code."""

from codegen.files import read_file
from codegen.utils import format_code


def test_async_methods_and_props():
    # Test that async methods return a promise

    for fname in ["_classes.py", "backends/wgpu_native/_api.py"]:
        code = format_code(read_file(fname), singleline=True)
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("def "):
                res_type = line.split("->")[-1].strip()
                if "_async(" in line:
                    assert res_type.startswith("GPUPromise")
                else:
                    assert "GPUPromise" not in line
