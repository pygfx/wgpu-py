# The sync API. Natively the backend polls; in Pyodide it suspends the Python
# stack with JSPI (pyodide.ffi.run_sync), so this needs e.g. runPythonAsync().
import struct
import wgpu
from wgpu.utils.compute import compute_with_buffers

SHADER = """
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read_write> b: array<i32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  b[id.x] = a[id.x] * a[id.x];
}
"""

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
print(
    "sync adapter:",
    adapter.info["vendor"],
    adapter.info["architecture"],
    adapter.info["backend_type"],
)
device = adapter.request_device_sync()
n = 20
data = struct.pack(f"{n}i", *range(n))
out = compute_with_buffers({0: data}, {1: (n, "i")}, SHADER, n=n)
result = list(out[1])
assert result == [i * i for i in range(n)], result
print("compute_with_buffers:", result[:6], "...")
buf = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.COPY_SRC)
back = device.queue.read_buffer(buf).cast("i")
assert list(back) == list(range(n))
print("COMPUTE SYNC OK")
