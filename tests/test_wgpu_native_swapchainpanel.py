"""Test presenting to a WinUI SwapChainPanel.

A real SwapChainPanel needs a WinUI app, so these tests use a minimal
stand-in for its ISwapChainPanelNative COM interface instead. That is
all that wgpu uses of the panel: it creates a swap chain for composition
and hands it to the panel with ISwapChainPanelNative::SetSwapChain.
"""

import gc
import sys
import ctypes

import wgpu
from pytest import raises, skip
from testutils import run_tests, can_use_wgpu_lib


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
if not sys.platform.startswith("win"):
    skip("SwapChainPanel is Windows only", allow_module_level=True)


HRESULT = ctypes.c_long
E_NOINTERFACE = ctypes.c_long(0x80004002).value

QueryInterface = ctypes.WINFUNCTYPE(
    HRESULT, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
)
AddRefOrRelease = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
SetSwapChain = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)


class FakeSwapChainPanelNative:
    """A COM object implementing ISwapChainPanelNative that records its calls."""

    class _Vtbl(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", QueryInterface),
            ("AddRef", AddRefOrRelease),
            ("Release", AddRefOrRelease),
            ("SetSwapChain", SetSwapChain),
        ]

    class _Object(ctypes.Structure):
        pass

    _Object._fields_ = [("vtbl", ctypes.POINTER(_Vtbl))]

    def __init__(self):
        self.refcount = 1
        self.swap_chains = []
        self._vtbl = self._Vtbl(
            QueryInterface(self._query_interface),
            AddRefOrRelease(self._add_ref),
            AddRefOrRelease(self._release),
            SetSwapChain(self._set_swap_chain),
        )
        self._object = self._Object(ctypes.pointer(self._vtbl))

    @property
    def pointer(self):
        return ctypes.addressof(self._object)

    def _query_interface(self, this, riid, ppv):
        ppv[0] = None
        return E_NOINTERFACE

    def _add_ref(self, this):
        self.refcount += 1
        return self.refcount

    def _release(self, this):
        self.refcount -= 1
        return self.refcount

    def _set_swap_chain(self, this, swap_chain):
        self.swap_chains.append(swap_chain)
        return 0


def get_adapter(d3d12):
    for adapter in wgpu.gpu.enumerate_adapters_sync():
        if (adapter.info["backend_type"] == "D3D12") == d3d12:
            return adapter
    skip("No D3D12 adapter" if d3d12 else "No adapter other than D3D12")


def test_swapchainpanel_present():
    adapter = get_adapter(d3d12=True)
    device = adapter.request_device_sync()
    panel = FakeSwapChainPanelNative()

    context = wgpu.gpu.get_canvas_context(
        {"platform": "swapchainpanel", "window": panel.pointer}
    )
    context.set_physical_size(64, 48)
    context.configure(device=device, format=context.get_preferred_format(adapter))
    assert panel.refcount > 1  # wgpu holds on to the panel

    for _ in range(3):
        texture = context.get_current_texture()
        assert texture.size[:2] == (64, 48)
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture.create_view(),
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (1, 0.5, 0, 1),
                }
            ]
        )
        render_pass.end()
        device.queue.submit([encoder.finish()])
        context.present()

    # The panel got a swap chain to show.
    assert len(panel.swap_chains) == 1
    assert panel.swap_chains[0]

    # And wgpu lets go of the panel.
    context.unconfigure()
    del context
    gc.collect()
    assert panel.refcount == 1


def test_swapchainpanel_compatible_adapter():
    get_adapter(d3d12=True)  # skip if there is none
    panel = FakeSwapChainPanelNative()

    context = wgpu.gpu.get_canvas_context(
        {"platform": "swapchainpanel", "window": panel.pointer}
    )
    for power_preference in ("low-power", "high-performance"):
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference, canvas=context
        )
        assert adapter.info["backend_type"] == "D3D12"


def test_swapchainpanel_needs_d3d12():
    adapter = get_adapter(d3d12=False)
    panel = FakeSwapChainPanelNative()

    context = wgpu.gpu.get_canvas_context(
        {"platform": "swapchainpanel", "window": panel.pointer}
    )
    with raises(RuntimeError, match="D3D12"):
        context.get_preferred_format(adapter)
    assert not panel.swap_chains


if __name__ == "__main__":
    run_tests(globals())
