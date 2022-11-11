# flake8: noqa
import gc
import os

os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

import psutil
import wgpu.backends.rs
from wgpu.gui.auto import WgpuCanvas, run

from triangle import main


p = psutil.Process()


def print_mem_usage(i):
    megs = p.memory_info().rss / 1024**2
    print(f"memory usage (round: {i}): {megs:.3f} MB")


if __name__ == "__main__":
    print_mem_usage(0)
    for i in range(10):
        canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle")
        device = main(canvas)
        run()
        gc.collect()
        print_mem_usage(i + 1)
