# flake8: noqa
import gc

import psutil

import wgpu.backends.rs
import wgpu


p = psutil.Process()


def print_mem_usage(i):
    megs = p.memory_info().rss / 1024**2
    print(f"memory usage (round: {i}): {megs:.3f} MB")


if __name__ == "__main__":
    print_mem_usage(0)
    for i in range(10):
        adapter = wgpu.request_adapter(canvas=None, power_preference="high-performance")
        device = adapter.request_device()
        device.destroy()
        del device
        wgpu.backends.rs.device_dropper.drop_all_pending()
        gc.collect()
        print_mem_usage(i + 1)
