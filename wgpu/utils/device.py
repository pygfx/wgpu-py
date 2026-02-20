import logging
from typing import Literal

import wgpu
from wgpu import enums
from wgpu._coreutils import CanvasLike
from wgpu import GPUAdapter, GPUDevice


__all__ = [
    "get_default_device",
    "preconfigure_default_device",
]

logger = logging.getLogger("wgpu")


class DefaultDeviceHelper:
    """Helper class to create a default device."""

    def __init__(self):
        self._caller_infos: dict = {}  # arg_name -> list of caller_info's.
        self._adapter_kwargs: dict = {}
        self._device_kwargs: dict = {}
        self._the_device: GPUDevice | None = None

    def preconfigure_default_device(
        self,
        caller_info: str,
        *,
        # Adapter arguments
        feature_level: Literal["core", "compatibility", None] = None,
        power_preference: enums.PowerPreferenceEnum | None = None,
        force_fallback_adapter: bool | None = None,
        canvas: CanvasLike | None = None,
        adapter: GPUAdapter | None = None,
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
            caller_info (str): A very brief description of the code that calls
                this, or the reason for calling it, e.g. a library name. For debugging only.
            feature_level (str): The feature level "core" (default) or "compatibility".
                This provides a way to opt into additional validation restrictions.
            power_preference (PowerPreference): "high-performance" or "low-power".
            force_fallback_adapter (bool): whether to use a (probably CPU-based) fallback adapter.
            canvas : The canvas or context that the adapter should be able to render to. This can typically
                be left to None. If given, it should be a ``GPUCanvasContext`` or ``RenderCanvas``.
            adapter (GPUAdapter): the adapter object to use to create the default device.
                This can be useful to target a specific GPU in a multi-GPU setting.
                Setting the adapter overrules all other adapter settings
                (feature_level, power_preference, force_fallback_adapter, canvas).
            label (str): A human-readable label for the device.
            required_features (list of str): the features (extensions) that you need.
            required_limits (dict): the various limits that you want to apply.

        """

        if not isinstance(caller_info, str):
            raise TypeError(
                f"preconfigure_default_device caller_info must be str, not {caller_info.__class__.__name__}"
            )

        if self._the_device is not None:
            raise RuntimeError(
                f"preconfigure_default_device ({caller_info}): default device cannot be configured after it is created; only call this at import-time."
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
            (ak, "adapter", adapter, GPUAdapter, None),
            (dk, "label", label, str, None),
            (dk, "required_features", required_features, set, enums.FeatureName),
            (dk, "required_limits", required_limits, dict, None),
        ]:
            if arg_value is None:
                continue

            if arg_type is not None:
                if not isinstance(arg_value, arg_type):
                    raise TypeError(
                        f"preconfigure_default_device ({caller_info}): {arg_name} must be a {arg_type.__name__}, but got {arg_value.__class__.__name__}."
                    )
            if arg_values is not None:
                if isinstance(arg_value, (set, dict)):
                    what, values = f"{arg_name} items", list(arg_value)
                else:
                    what, values = arg_name, [arg_value]
                for value in values:
                    if value not in arg_values:
                        raise ValueError(
                            f"preconfigure_default_device ({caller_info}): {what} must be a one of {set(arg_values)}, but got {value!r}."
                        )
            if isinstance(arg_value, set):
                cur_value = arg_dict.setdefault(arg_name, set())
                cur_value.update(arg_value)
            elif isinstance(arg_value, dict):
                cur_value = arg_dict.setdefault(arg_name, dict())
                for key, val in arg_value.items():
                    cur_value[key] = min(val, cur_value.get(key, val))
            else:
                cur_value = arg_dict.get(arg_name)
                if cur_value is not None and cur_value != arg_value:
                    caller_stack = self._caller_infos.get(arg_name, [])
                    prev_caller = caller_stack[-1] if caller_stack else "unknown"
                    logger.warning(
                        f"preconfigure_default_device ({caller_info}): {arg_name} set to {arg_value!r} overrides earlier set {cur_value!r} by {prev_caller!r}."
                    )
                arg_dict[arg_name] = arg_value

            # Register caller for this arg
            self._caller_infos.setdefault(arg_name, []).append(caller_info)

            # Handle that adapter overrides other adapter args.
            if arg_name == "adapter":
                for key, cur_value in list(arg_dict.items()):
                    if key != "adapter":
                        caller_stack = self._caller_infos.get(key, [])
                        prev_caller = caller_stack[-1] if caller_stack else "unknown"
                        logger.warning(
                            f"preconfigure_default_device ({caller_info}): setting adapter overrides {key}={cur_value!r} set by {prev_caller!r}."
                        )
                        arg_dict.pop(key, None)
            elif arg_dict.get("adapter") is not None:
                caller_stack = self._caller_infos.get("adapter", [])
                prev_caller = caller_stack[-1] if caller_stack else "unknown"
                logger.warning(
                    f"preconfigure_default_device ({caller_info}): setting {arg_name} is ignored because adapter is set by {prev_caller!r}."
                )
                arg_dict.pop(arg_name, None)

    def get_default_device(self) -> GPUDevice:
        """Get the default ``GPUDevice`` instance.

        The default device is a global/shared device. It is generally
        recommended to use this device; different parts of a running program can
        only share objects like textures and buffers when they use the same
        device.

        The default device can be configured at import-time using ``preconfigure_default_device()``.
        """
        if self._the_device is None:
            adapter: GPUAdapter = self._adapter_kwargs.pop("adapter", None)
            if adapter is None:
                adapter = wgpu.gpu.request_adapter_sync(**self._adapter_kwargs)
            self._the_device = adapter.request_device_sync(**self._device_kwargs)
            self._adapter_kwargs.clear()
            self._device_kwargs.clear()
        return self._the_device


helper = DefaultDeviceHelper()

# Public API functions
preconfigure_default_device = helper.preconfigure_default_device
get_default_device = helper.get_default_device
