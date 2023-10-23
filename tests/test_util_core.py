import wgpu
from wgpu._coreutils import error_message_hash, str_flag_to_int, _flag_cache
from testutils import run_tests


def test_error_message_hash():
    text1 = """In wgpuRenderPassEncoderEnd
    In a pass parameter
      note: command buffer = `<CommandBuffer-(0, 8, Metal)>`
    The color attachment at index 0's texture view is not renderable:
    """

    text2 = """In wgpuRenderPassEncoderEnd
    In a pass parameter
      note: command buffer = `<CommandBuffer-(2, 8, Metal)>`
    The color attachment at index 0's texture view is not renderable:
    """

    text3 = """In wgpuRenderPassEncoderEnd
    In a pass parameter BLABLA
      note: command buffer = `<CommandBuffer-(2, 8, Metal)>`
    The color attachment at index 0's texture view is not renderable:
    """

    assert error_message_hash(text1) == error_message_hash(text2)
    assert error_message_hash(text1) != error_message_hash(text3)


def test_str_flag_to_int():
    versions = [
        "UNIFORM|VERTEX",
        "UNIFORM | VERTEX",
        "VERTEX | UNIFORM",
        "VERTEX|  UNIFORM",
    ]

    flags = [str_flag_to_int(wgpu.BufferUsage, v) for v in versions]

    for flag in flags:
        assert flag == flags[0]

    for v in versions:
        assert f"BufferUsage.{v}" in _flag_cache


if __name__ == "__main__":
    run_tests(globals())
