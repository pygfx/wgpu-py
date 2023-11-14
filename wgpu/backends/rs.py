# Termporaty alias for backwards compatibility.

from .wgpu_native import gpu  # noqa

_deprecation_msg = """
WARNING: wgpu.backends.rs is deprecated. Instead you can use:
- import wgpu.backends.wgpu_native to use the backend by its new name.
- import wgpu.backends.auto to do the same, but simpler and more future proof.
- simply use wgpu.gpu.request_adapter() to auto-load the backend.
""".strip()

print(_deprecation_msg)
