""" Test some parts of rsbackend.py, and implicitly tests hparser.py.
"""

from codegen.rspatcher import patch_rs_backend


def dedent(code):
    return code.replace("\n    ", "\n")


def test_patch_functions():

    code1 = """
    lib.wgpu_adapter_request_device(1, 2, 3)
    lib.wgpu_foo_bar(1, 2, 3)
    """

    code2 = patch_rs_backend(dedent(code1))

    # All original lines are there
    assert all(line[4:] in code2 for line in code1 if line.strip())

    # But also an annotation
    assert "WGPUAdapterId adapter_id, const WGPUDeviceDescriptor" in code2
    # And a notification that foo_bar is unknown
    assert code2.count("# FIXME:") == 1
    assert code2.count("foo_bar") == 2


def test_patch_structs():

    # Check simple struct
    code1 = """
    struct = new_struct_p(
        "WGPUBufferDescriptor *",
        label=c_label,
        size=size,
        usage=usage,
    )
    """
    code2 = patch_rs_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "label: WGPULabel, size: WGPU" in code2
    assert "# FIXME:" not in code2
    assert code2 == patch_rs_backend(code2)  # Don't stack comments

    # Check, but now using not-pointer
    code1 = """
    struct = new_struct(
        "WGPUBufferDescriptor",
        label=c_label,
        size=size,
        usage=usage,
    )
    """
    code2 = patch_rs_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "label: WGPULabel, size: WGPU" in code2
    assert "# FIXME:" not in code2

    # Fail
    code1 = 'struct = new_struct("WGPUBufferDescriptor *",label=c_label,size=size,usage=usage,)'
    code2 = patch_rs_backend(dedent(code1))
    assert "# FIXME:" in code2
    assert code2 == patch_rs_backend(code2)  # Don't stack comments

    # Fail
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor",label=c_label,size=size,usage=usage,)'
    code2 = patch_rs_backend(dedent(code1))
    assert "# FIXME:" in code2
    assert code2 == patch_rs_backend(code2)  # Don't stack comments

    # Missing values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,size=size,)'
    code2 = patch_rs_backend(dedent(code1))
    assert "label: WGPULabel, size: WGPU" in code2
    assert "# FIXME:" not in code2
    assert "usage" in code2  # comment added
    assert code2 == patch_rs_backend(code2)  # Don't stack comments

    # Too many values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,foo=size,)'
    code2 = patch_rs_backend(dedent(code1))
    assert "label: WGPULabel, size: WGPU" in code2
    assert "# FIXME: unknown" in code2
    assert code2 == patch_rs_backend(code2)  # Don't stack comments


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
