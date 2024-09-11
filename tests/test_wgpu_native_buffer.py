import random
import ctypes
import sys

import wgpu.utils
import numpy as np

from testutils import run_tests, can_use_wgpu_lib, iters_equal
from pytest import mark, raises


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init1():
    # Initializing a buffer with data

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    assert repr(device).startswith("<wgpu.backends.wgpu_native.GPUDevice ")

    # Create buffer. COPY_SRC is needed to read the buffer via the queue.
    buf = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.COPY_SRC)

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # ---  also read via mapped data

    # Create buffer. MAP_READ is needed to read the buffer via the queue.
    buf = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.MAP_READ)

    wgpu.backends.wgpu_native._api.libf.wgpuDevicePoll(
        buf._device._internal, True, wgpu.backends.wgpu_native.ffi.NULL
    )

    # Download from buffer to CPU
    buf.map(wgpu.MapMode.READ)
    wgpu.backends.wgpu_native._api.libf.wgpuDevicePoll(
        buf._device._internal, True, wgpu.backends.wgpu_native.ffi.NULL
    )

    data2 = buf.read_mapped()
    buf.unmap()
    print(data2.tobytes())
    assert data1 == data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init2():
    # Initializing a buffer as mapped, to directly set the data

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # Create buffer.
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_SRC, mapped_at_creation=True
    )
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # --- also read via mapped data

    # Create buffer.
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_READ, mapped_at_creation=True
    )
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped()
    buf.unmap()
    print(data2.tobytes())
    assert data1 == data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init3():
    # Initializing an empty buffer, then writing to it, then reading back

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # Option 1: write via queue (i.e. temp buffer), read via queue

    # Create buffer
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # Option 2: Write via mapped data, read via queue

    # Create buffer
    buf = device.create_buffer(size=len(data1), usage="MAP_WRITE | COPY_SRC")

    # Write data to it
    buf.map("write")
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # Option 3: Write via queue, read via mapped data

    buf = device.create_buffer(size=len(data1), usage=" MAP_READ | COPY_DST ")

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped()
    buf.unmap()
    assert data1 == data2

    # Option 4: Write via mapped data, read via mapped data

    # Not actually an option
    with raises(wgpu.GPUValidationError):
        buf = device.create_buffer(size=len(data1), usage="MAP_READ |MAP_WRITE")


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_writes1():
    # The inefficient way

    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    # Write in parts
    for i in range(4):
        buf.map("write")
        buf.write_mapped(f"{i+1}".encode() * 8, i * 8)
        buf.unmap()

    # Download from buffer to CPU
    data = device.queue.read_buffer(buf)
    assert data == b"11111111222222223333333344444444"

    # Also in parts
    for i in range(4):
        data = device.queue.read_buffer(buf, i * 8, size=8)
        assert data == f"{i+1}".encode() * 8


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_writes2():
    # The efficient way

    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    # Write in parts
    buf.map("write")
    for i in range(4):
        buf.write_mapped(f"{i+1}".encode() * 8, i * 8)
    buf.unmap()

    # Download from buffer to CPU
    data = device.queue.read_buffer(buf)
    assert data == b"11111111222222223333333344444444"

    # Also in parts
    for i in range(4):
        data = device.queue.read_buffer(buf, i * 8, size=8)
        assert data == f"{i+1}".encode() * 8


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_reads():
    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Write using the queue. Do in parts, to touch those offsets too
    for i in range(4):
        device.queue.write_buffer(buf, i * 8, f"{i+1}".encode() * 8)

    # Read in parts, the inefficient way
    for i in range(4):
        buf.map("read")
        data = buf.read_mapped(i * 8, 8)
        assert data == f"{i+1}".encode() * 8
        buf.unmap()

    # Read in parts, the efficient way
    buf.map("read")
    for i in range(4):
        data = buf.read_mapped(i * 8, 8)
        assert data == f"{i+1}".encode() * 8
    buf.unmap()


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_mapping_fails():
    device = wgpu.utils.get_default_device()
    data = b"12345678"

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    with raises(RuntimeError):
        buf.write_mapped(data)  # Not mapped
    with raises(RuntimeError):
        buf.read_mapped()  # Not mapped

    with raises(ValueError):
        buf.map("boo")  # Invalid map mode

    buf.map("write", 0, 28)

    with raises(RuntimeError):
        buf.map("write")  # Cannot map twice

    with raises(RuntimeError):
        buf.map("read")  # Cannot map twice

    with raises(RuntimeError):
        buf.read_mapped()  # Not mapped in read mode

    # Ok
    buf.write_mapped(data)
    buf.write_mapped(data, 0)
    buf.write_mapped(data, 8)
    buf.write_mapped(data, 16)

    # Fail
    with raises(ValueError):
        buf.write_mapped(data, -1)  # not neg
    with raises(ValueError):
        buf.write_mapped(data, -8)  # not neg
    with raises(ValueError):
        buf.write_mapped(data, 6)  # not multiple of eight

    # Ok
    buf.write_mapped(b"1" * 4)
    buf.write_mapped(b"1" * 8)
    buf.write_mapped(b"1" * 28)
    buf.write_mapped(b"1" * 12, 0)
    buf.write_mapped(b"1" * 12, 8)

    with raises(ValueError):
        buf.write_mapped(b"")  # not empty
    with raises(ValueError):
        buf.write_mapped(b"1" * 64)  # too large for buffer
    with raises(ValueError):
        buf.write_mapped(b"1" * 32)  # too large for mapped range
    with raises(ValueError):
        buf.write_mapped(b"1" * 3)  # not multiple of 4
    with raises(ValueError):
        buf.write_mapped(b"1" * 6)  # not multiple of 4
    with raises(ValueError):
        buf.write_mapped(b"1" * 9)  # not multiple of 4

    # Can unmap multiple times though!
    buf.unmap()

    with raises(RuntimeError):
        buf.unmap()  # Cannot unmap when not mapped

    # Create buffer in read mode ...

    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    with raises(RuntimeError):
        buf.write_mapped(data)  # not mapped

    buf.map("read", 8, 20)

    with raises(RuntimeError):
        buf.map("read")  # Cannot map twice

    with raises(RuntimeError):
        buf.map("write")  # Cannot map twice

    with raises(RuntimeError):
        buf.write_mapped(data)  # not mapped in write mode

    # Ok
    assert len(buf.read_mapped()) == 20

    # Fail
    with raises(ValueError):
        buf.read_mapped(0, 64)  # read beyond buffer size
    with raises(ValueError):
        buf.read_mapped(0, 32)  # read beyond mapped range

    buf.unmap()

    with raises(RuntimeError):
        buf.unmap()  # Cannot unmap when not mapped


def test_buffer_read_no_copy():
    device = wgpu.utils.get_default_device()
    data1 = b"12345678" * 2

    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped(copy=False)
    data3 = buf.read_mapped(0, 8, copy=False)
    data4 = buf.read_mapped(8, 8, copy=False)

    assert data2 == data1
    assert data3 == data1[:8]
    assert data4 == data1[8:]

    # Can access the arrays
    _ = data2[0], data3[0], data4[0]

    # But cannot write to memory intended for reading
    if sys.version_info >= (3, 8):  # no memoryview.toreadonly on 3.7 and below
        with raises(TypeError):
            data2[0] = 1
        with raises(TypeError):
            data3[0] = 1
        with raises(TypeError):
            data4[0] = 1

    buf.unmap()

    # The memoryview is invalidated when the buffer unmapped.
    # Note that this unfortunately does *not* hold for views on these arrays.
    with raises(ValueError):
        data2[0]
    with raises(ValueError):
        data3[0]
    with raises(ValueError):
        data4[0]

    with raises(ValueError):
        data2[0] = 1
    with raises(ValueError):
        data3[0] = 1
    with raises(ValueError):
        data4[0] = 1


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_clear_buffer():
    data = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # length 52
    length = len(data)

    # Prep
    device = wgpu.utils.get_default_device()
    buf = device.create_buffer(size=length, usage="COPY_DST|COPY_SRC")

    # Helper function that writes "data" to the buffer, clears part of it, and then
    # returns the resulting value in the buffer.
    def run_clear_buffer(*args):
        device.queue.write_buffer(buf, 0, data)
        command_encoder = device.create_command_encoder()
        command_encoder.clear_buffer(buf, *args)
        device.queue.submit([command_encoder.finish()])
        result = device.queue.read_buffer(buf)
        return bytes(result)

    assert run_clear_buffer(8, 12) == data[:8] + bytes(12) + data[20:]
    assert run_clear_buffer(8) == data[:8] + bytes(length - 8)
    assert run_clear_buffer() == bytes(length)

    with raises(ValueError):
        run_clear_buffer(10)  # offset not a multiple of 4
    with raises(ValueError):
        run_clear_buffer(-10)  # offset negative
    with raises(ValueError):
        run_clear_buffer(12, 30)  # size not a multiple of 4
    with raises(ValueError):
        run_clear_buffer(12, length)  # size too large, given offset


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer1():
    device = wgpu.utils.get_default_device()

    data1 = memoryview(np.random.random(size=100).astype(np.float32))

    # Create buffer
    buf4 = device.create_buffer(
        size=data1.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Upload from CPU to buffer
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, data1)
    device.queue.submit([])

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf4).cast("f")
    assert data1 == data2

    # Yes, you can compare memoryviews! Check this:
    data1[0] += 1
    assert data1 != data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer2():
    device = wgpu.utils.get_default_device()

    nx, ny, nz = 100, 1, 1
    data0 = (ctypes.c_float * 100)(*[random.random() for i in range(nx * ny * nz)])
    data1 = (ctypes.c_float * 100)()
    nbytes = ctypes.sizeof(data1)

    # Create buffer
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    for i in range(len(data1)):
        data1[i] = data0[i]

    # Upload from CPU to buffer
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, data1)

    # We swipe the data. You could also think that we passed something into
    # write_buffer without holding a reference to it. Anyway, write_buffer
    # seems to copy the data at the moment it is called.
    for i in range(len(data1)):
        data1[i] = 1

    device.queue.submit([])

    # Download from buffer to CPU
    data2 = data1.__class__.from_buffer(device.queue.read_buffer(buf4))
    assert iters_equal(data0, data2)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer3():
    device = wgpu.utils.get_default_device()
    nbytes = 12

    # Create buffer
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Upload from CPU to buffer, using bytes
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, b"abcdefghijkl", 0, nbytes)
    device.queue.submit([])

    # Download from buffer to CPU
    assert device.queue.read_buffer(buf4).tobytes() == b"abcdefghijkl"


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_map_read_and_write():
    # Do a mini round-trip using mapped buffers

    device = wgpu.utils.get_default_device()
    nbytes = 12

    # Create buffers
    buf1 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.MAP_WRITE
    )
    buf2 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    # Upload
    data1 = b"abcdefghijkl"
    buf1.map("write")
    buf1.write_mapped(data1)
    buf1.unmap()

    # Copy
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buf1, 0, buf2, 0, nbytes)
    device.queue.submit([command_encoder.finish()])

    # Download
    buf2.map("read")
    data2 = buf2.read_mapped()
    buf2.unmap()
    assert data1 == data2


if __name__ == "__main__":
    run_tests(globals())
