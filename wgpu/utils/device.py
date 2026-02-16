import sys
import logging
from typing import Literal

import wgpu
from wgpu import enums
from wgpu._coreutils import CanvasLike
from wgpu import GPUPromise, GPUAdapter, GPUDevice


__all__ = [
    "get_default_device",
    "preconfigure_default_device",
    "request_default_device",
]

logger = logging.getLogger("wgpu")


class DefaultDeviceHelper:
    """Helper class to create a default device."""

    def __init__(self):
        self._adapter_kwargs: dict = {}
        self._device_kwargs: dict = {}
        self._adapter_promise: GPUPromise | None = None
        self._device_promise: GPUPromise | None = None
        self._the_promise: GPUPromise | None = None
        self._the_device: GPUDevice | None = None

    def _create_the_promise(self) -> GPUPromise:
        if self._the_promise is None:
            from wgpu.backends.auto import gpu

            module = sys.modules[gpu.__module__]
            self._the_promise = module.GPUPromise("default_device", None)

        return self._the_promise

    def _request_adapter(self) -> GPUPromise:
        if self._adapter_promise is None:
            self._adapter_promise = wgpu.gpu.request_adapter_async(
                **self._adapter_kwargs
            )
            self._adapter_kwargs.clear()

        return self._adapter_promise

    def _request_device(self, adapter: GPUAdapter) -> GPUPromise:
        if self._device_promise is None:
            self._device_promise = adapter.request_device_async(**self._device_kwargs)
            self._device_kwargs.clear()

        return self._device_promise

    def _set_the_device(self, device: GPUDevice) -> None:
        if self._the_device is None:
            self._create_the_promise()
            self._the_promise._wgpu_set_input(device)
            self._the_device = device

    def preconfigure_default_device(
        self,
        context: str,
        *,
        # Adapter arguments
        feature_level: Literal["core", "compatibility", None] = None,
        power_preference: enums.PowerPreferenceEnum | None = None,
        force_fallback_adapter: bool | None = None,
        canvas: CanvasLike | None = None,
        # Device arguments
        label: str | None = None,
        required_features: set[enums.FeatureNameEnum] | None = None,
        required_limits: dict[str, int | None] | None = None,
        # default_queue: structs.QueueDescriptorStruct | None = None,
    ) -> None:
        """Configure the default ``GPUDevice`` before it is created.

        This function can only be called at import-time or at the top of your
        script. It is an error to call this once the default device has been
        created.

        This function can be called multiple times, e.g. different libraries that
        use wgpu can each require the features they need. For required features
        the union of set features is used. For required limits the minimum of
        each set limit is used. For the other arguments, the last set value is
        used, and a warning is logged when a value is overriden.

        Arguments:
            context (str): A very brief description for the code that calls
                this, or the reason for calling it, e.g. a library name. For debugging only.
            feature_level (str): The feature level "core" (default) or "compatibility".
                This provides a way to opt into additional validation restrictions.
            power_preference (PowerPreference): "high-performance" or "low-power".
            force_fallback_adapter (bool): whether to use a (probably CPU-based) fallback adapter.
            canvas : The canvas or context that the adapter should be able to render to. This can typically
                be left to None. If given, it should be a ``GPUCanvasContext`` or ``RenderCanvas``.
            label (str): A human-readable label for the device.
            required_features (list of str): the features (extensions) that you need.
            required_limits (dict): the various limits that you want to apply.

        """

        if not isinstance(context, str):
            raise TypeError(
                f"preconfigure_default_device context must be str, not {context.__class__.__name__}"
            )

        if self._the_promise is not None:
            raise RuntimeError(
                f"preconfigure_default_device ({context}): default device cannot be configured after it is requested/created; only call this at import-time."
            )

        # We want to do a good job validating inputs here, because the global device
        # can be requested from multiple places, and if an invalid arg is set from
        # one of these, and we use it to request the adapter, it's hard to establish
        # where the invalid arg was set from.

        if required_features is not None:
            required_features = set(required_features)

        ak, dk = self._adapter_kwargs, self._device_kwargs

        for arg_dict, arg_name, arg_value, arg_type, arg_values in [
            (ak, "feature_level", feature_level, str, ("core", "compatibility")),
            (ak, "power_preference", power_preference, str, enums.PowerPreference),
            (ak, "force_fallback_adapter", force_fallback_adapter, bool, None),
            (ak, "canvas", canvas, None, None),
            (dk, "label", label, str, None),
            (dk, "required_features", required_features, set, enums.FeatureName),
            (dk, "required_limits", required_limits, dict, None),
        ]:
            if arg_value is not None:
                if arg_type is not None:
                    if not isinstance(arg_value, arg_type):
                        raise TypeError(
                            f"preconfigure_default_device ({context}): {arg_name} must be a {arg_type.__name__}, but got {arg_value.__class__.__name__}."
                        )
                if arg_values is not None:
                    if isinstance(arg_value, (set, dict)):
                        what, values = f"{arg_name} items", list(arg_value)
                    else:
                        what, values = arg_name, [arg_value]
                    for value in values:
                        if value not in arg_values:
                            raise ValueError(
                                f"preconfigure_default_device ({context}): {what} must be a one of {set(arg_values)}, but got {value!r}."
                            )
                if isinstance(arg_value, set):
                    cur_value = arg_dict.setdefault(arg_name, set())
                    cur_value.update(arg_value)
                elif isinstance(arg_value, dict):
                    cur_value = arg_dict.setdefault(arg_name, dict())
                    for key, val in arg_value.items():
                        cur_value[key] = min(val, cur_value.get(key, val))
                else:
                    cur_value = arg_dict.get(arg_name, None)
                    if cur_value is not None and cur_value != arg_value:
                        logger.warning(
                            f"preconfigure_default_device ({context}): {arg_name} overrides earlier set {cur_value!r} with {arg_value!r}."
                        )
                    arg_dict[arg_name] = arg_value

    def request_default_device(self) -> GPUPromise[GPUDevice]:
        """Request the default ``GPUDevice`` instance, returns a promise.

        The default device is a global/shared device that is generally recommended;
        Different parts of a running program that use the same device and share
        objects like textures and buffers.

        The default device can be configured at import time using ``preconfigure_default_device()``.
        """

        if self._the_promise is None:
            self._create_the_promise()
            adapter_promise = self._request_adapter()
            device_promise = adapter_promise.then(self._request_device)
            device_promise.then(self._set_the_device)

        return self._the_promise

    def get_default_device(self) -> GPUDevice:
        """Get the default ``GPUDevice`` instance.

        This is a sync version of ``request_default_device``.
        It is nice and simple, but if you want your code to run in the browser (on Pyodide/PyScript)
        you should consider ``request_default_device`` instead.

        The default device can be configured at import time using ``preconfigure_default_device()``.
        """
        if self._the_device is None:
            self._create_the_promise()
            adapter_promise = self._request_adapter()
            adapter = adapter_promise.sync_wait()
            device_promise = self._request_device(adapter)
            device = device_promise.sync_wait()
            self._set_the_device(device)

        return self._the_device


# The helper is an implementation detail, intended to keep code clean and allow unit tests.
helper = DefaultDeviceHelper()

# Public API functions
preconfigure_default_device = helper.preconfigure_default_device
request_default_device = helper.request_default_device
get_default_device = helper.get_default_device
