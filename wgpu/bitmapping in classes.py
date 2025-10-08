    _copy_buffer = None, 0

    def _present_bitmap(self):

        ...

        source = {
            "texture": texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
        }

        ori_stride = bytes_per_pixel * size[0]
        extra_stride = (256 - ori_stride % 256) % 256
        full_stride = ori_stride + extra_stride

        data_length = full_stride * size[1] * size[2]

        # Create temporary buffer
        copy_buffer, time_since_size_ok = self._copy_buffer
        if copy_buffer is None:
            pass  # No buffer
        elif copy_buffer.size < data_length:
            copy_buffer = None  # Buffer too small
        elif copy_buffer.size < data_length * 4:
            self._copy_buffer = copy_buffer, time.time()  # Bufer size ok
        elif time.time() - time_since_size_ok > 5.0:
            copy_buffer = None  # Too large too long
        if copy_buffer is None:
            buffer_size = data_length
            buffer_size += (4096 - buffer_size % 4096) % 4096
            buf_usage = flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ
            copy_buffer = device._create_buffer(
                "copy-buffer", buffer_size, buf_usage, False
            )
            self._copy_buffer = copy_buffer, time.time()

        destination = {
            "buffer": copy_buffer,
            "offset": 0,
            "bytes_per_row": full_stride,  # or WGPU_COPY_STRIDE_UNDEFINED ?
            "rows_per_image": size[1],
        }

        # Copy data to temp buffer
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(source, destination, size)
        command_buffer = encoder.finish()
        device.queue.submit([command_buffer])

        awaitable = copy_buffer.map_async("READ_NOSYNC", 0, data_length)

        # Download from mappable buffer
        # Because we use `copy=False``, we *must* copy the data.
        if copy_buffer.map_state == "pending":
            awaitable.sync_wait()
        mapped_data = copy_buffer.read_mapped(copy=False)

        data_length2 = ori_stride * size[1] * size[2]

        # Copy the data
        if extra_stride:
            # Copy per row
            data = memoryview(bytearray(data_length2)).cast(mapped_data.format)
            i_start = 0
            for i in range(size[1] * size[2]):
                row = mapped_data[i * full_stride : i * full_stride + ori_stride]
                data[i_start : i_start + ori_stride] = row
                i_start += ori_stride
        else:
            # Copy as a whole
            data = memoryview(bytearray(mapped_data)).cast(mapped_data.format)

        # Alternative copy solution using Numpy.
        # I expected this to be faster, but does not really seem to be. Seems not worth it
        # since we technically don't depend on Numpy. Leaving here for reference.
        # import numpy as np
        # mapped_data = np.asarray(mapped_data)[:data_length]
        # data = np.empty(data_length2, dtype=mapped_data.dtype)
        # mapped_data.shape = -1, full_stride
        # data.shape = -1, ori_stride
        # data[:] = mapped_data[:, :ori_stride]
        # data.shape = -1
        # data = memoryview(data)

        # Since we use read_mapped(copy=False), we must unmap it *after* we've copied the data.
        copy_buffer.unmap()