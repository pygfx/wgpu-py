""" Test some parts of rsbackend.py, and implicitly tests hparser.py.
"""

from codegen.wgpu_native_patcher import patch_wgpu_native_backend


def dedent(code):
    return code.replace("\n    ", "\n")


def test_patch_functions():
    code1 = """
    libf.wgpuAdapterRequestDevice(1, 2, 3)
    libf.wgpuFooBar(1, 2, 3)
    """

    code2 = patch_wgpu_native_backend(dedent(code1))

    # All original lines are there
    assert all(line[4:] in code2 for line in code1 if line.strip())

    # But also an annotation
    assert "WGPUAdapter adapter, WGPUDeviceDescriptor" in code2
    # And a notification that foo_bar is unknown
    assert code2.count("# FIXME:") == 1
    assert code2.count("FooBar") == 2


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
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "usage: WGPUBufferUsageFlags/int" in code2
    assert "size: int" in code2
    assert "# FIXME:" not in code2
    assert code2 == patch_wgpu_native_backend(code2)  # Don't stack comments

    # Check, but now using not-pointer
    code1 = """
    struct = new_struct(
        "WGPUBufferDescriptor",
        label=c_label,
        size=size,
        usage=usage,
    )
    """
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "usage: WGPUBufferUsageFlags/int" in code2
    assert "size: int" in code2
    assert "# FIXME:" not in code2

    # Fail
    code1 = 'struct = new_struct("WGPUBufferDescriptor *",label=c_label,size=size,usage=usage,)'
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert "# FIXME:" in code2
    assert code2 == patch_wgpu_native_backend(code2)  # Don't stack comments

    # Fail
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor",label=c_label,size=size,usage=usage,)'
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert "# FIXME:" in code2
    assert code2 == patch_wgpu_native_backend(code2)  # Don't stack comments

    # Missing values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,size=size,)'
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert "usage: WGPUBufferUsageFlags/int" in code2
    assert "# FIXME:" not in code2
    assert "usage" in code2  # comment added
    assert code2 == patch_wgpu_native_backend(code2)  # Don't stack comments

    # Too many values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,foo=size,)'
    code2 = patch_wgpu_native_backend(dedent(code1))
    assert "usage: WGPUBufferUsageFlags/int" in code2
    assert "# FIXME: unknown" in code2
    assert code2 == patch_wgpu_native_backend(code2)  # Don't stack comments


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
