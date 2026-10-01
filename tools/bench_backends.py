"""
Benchmark the per-call overhead of the wgpu-py API for a given backend.

This measures the cost of common calls *through the public wgpu-py API*
(i.e. including all Python-side logic of the backend), using cheap calls so
that the Python -> native overhead dominates. Run it once per backend::

    WGPUPY_BACKEND=wgpu_native python tools/bench_backends.py
    WGPUPY_BACKEND=dawn python tools/bench_backends.py

Or run ``python tools/bench_backends.py --all`` to run all available backends
in subprocesses and print a markdown table.
"""

import os
import sys
import json
import time
import subprocess


REPEAT = 5


def best_ns(func, n):
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        func(n)
        times.append(time.perf_counter() - t0)
    return min(times) / n * 1e9


def run_benchmarks():
    import numpy as np
    import wgpu

    device = wgpu.utils.get_default_device()
    adapter_summary = device.adapter.summary

    shader = device.create_shader_module(
        code="""
        @group(0) @binding(0) var<uniform> u: vec4<f32>;
        @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
            return vec4<f32>(f32(i % 2u), f32(i / 2u), 0.0, 1.0) * u.x;
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> { return u; }
        @group(0) @binding(1) var<storage, read_write> data: array<u32>;
        @compute @workgroup_size(1) fn cs_main() { data[0] = data[0] + 1u; }
        """
    )
    ubuf = device.create_buffer(
        size=256, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    sbuf = device.create_buffer(
        size=256, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX
                | wgpu.ShaderStage.FRAGMENT
                | wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
        ]
    )
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": ubuf}},
            {"binding": 1, "resource": {"buffer": sbuf}},
        ],
    )
    render_pipeline = device.create_render_pipeline(
        layout=layout,
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": "rgba8unorm"}],
        },
    )
    compute_pipeline = device.create_compute_pipeline(
        layout=layout, compute={"module": shader, "entry_point": "cs_main"}
    )
    texture = device.create_texture(
        size=(64, 64, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    view = texture.create_view()
    color_attachments = [
        {
            "view": view,
            "load_op": "clear",
            "store_op": "store",
            "clear_value": (0, 0, 0, 1),
        }
    ]
    data64 = np.zeros(16, np.float32)

    results = {}

    # Recording calls inside a pass. We start a new pass for each run, and
    # record many calls into it (not submitted).
    def in_render_pass(body):
        def run(n):
            encoder = device.create_command_encoder()
            rp = encoder.begin_render_pass(color_attachments=color_attachments)
            rp.set_pipeline(render_pipeline)
            rp.set_bind_group(0, bind_group)
            body(rp, n)
            rp.end()
            encoder.finish()

        return run

    def draw(rp, n):
        f = rp.draw
        for _ in range(n):
            f(3)

    def set_bind_group(rp, n):
        f = rp.set_bind_group
        for _ in range(n):
            f(0, bind_group)

    def set_viewport(rp, n):
        f = rp.set_viewport
        for _ in range(n):
            f(0, 0, 64, 64, 0, 1)

    def dispatch(n):
        encoder = device.create_command_encoder()
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(compute_pipeline)
        cp.set_bind_group(0, bind_group)
        f = cp.dispatch_workgroups
        for _ in range(n):
            f(1)
        cp.end()
        encoder.finish()

    def write_buffer(n):
        f = device.queue.write_buffer
        for _ in range(n):
            f(ubuf, 0, data64)
        device.queue.submit([])

    def frame(n):
        queue = device.queue
        for i in range(n):
            encoder = device.create_command_encoder()
            rp = encoder.begin_render_pass(color_attachments=color_attachments)
            rp.set_pipeline(render_pipeline)
            rp.set_bind_group(0, bind_group)
            for _ in range(10):
                rp.draw(3)
            rp.end()
            queue.submit([encoder.finish()])
            if i % 100 == 99:
                queue.on_submitted_work_done_sync()
        queue.on_submitted_work_done_sync()

    def create_bind_group(n):
        entries = [
            {"binding": 0, "resource": {"buffer": ubuf}},
            {"binding": 1, "resource": {"buffer": sbuf}},
        ]
        for _ in range(n):
            device.create_bind_group(layout=bgl, entries=entries)

    results["render_pass.draw(3)"] = best_ns(in_render_pass(draw), 100_000)
    results["render_pass.set_bind_group(0, bg)"] = best_ns(
        in_render_pass(set_bind_group), 100_000
    )
    results["render_pass.set_viewport(...)"] = best_ns(
        in_render_pass(set_viewport), 100_000
    )
    results["compute_pass.dispatch_workgroups(1)"] = best_ns(dispatch, 100_000)
    results["queue.write_buffer(64 bytes)"] = best_ns(write_buffer, 20_000)
    results["device.create_bind_group(2 entries)"] = best_ns(create_bind_group, 5_000)
    results["frame: encoder+pass+10 draws+submit"] = best_ns(frame, 2_000)

    return {"adapter": adapter_summary, "results": results}


def main():
    if "--all" not in sys.argv:
        out = run_benchmarks()
        print(json.dumps(out))
        return

    all_results = {}
    for backend in ("wgpu_native", "dawn"):
        env = dict(os.environ, WGPUPY_BACKEND=backend)
        p = subprocess.run(
            [sys.executable, __file__], env=env, capture_output=True, text=True
        )
        lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
        if p.returncode or not lines:
            print(f"Backend {backend} failed:\n{p.stderr[-2000:]}")
            continue
        all_results[backend] = json.loads(lines[-1])

    backends = list(all_results)
    print("| call (ns per call, lower is better) | " + " | ".join(backends) + " |")
    print("|---|" + "---:|" * len(backends))
    keys = next(iter(all_results.values()))["results"].keys() if backends else []
    for key in keys:
        vals = " | ".join(f"{all_results[b]['results'][key]:,.0f}" for b in backends)
        print(f"| {key} | {vals} |")
    for b in backends:
        print(f"* {b}: {all_results[b]['adapter']}")


if __name__ == "__main__":
    main()
