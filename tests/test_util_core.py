from wgpu._coreutils import error_message_hash
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


if __name__ == "__main__":
    run_tests(globals())
