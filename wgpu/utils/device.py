import logging
from typing import Sequence

import wgpu
from wgpu import enums
from wgpu._coreutils import CanvasLike

logger = logging.getLogger("wgpu")

_default_device = None


def get_default_device():
    """Get a wgpu device object. If this succeeds, it's likely that
    the WGPU lib is usable on this system. If not, this call will
    probably exit (Rust panic). When called multiple times,
    returns the same global device object (useful for e.g. unit tests).
    """
    global _default_device

    if _default_device is None:
        import wgpu.backends.auto

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _default_device = adapter.request_device_sync()
    return _default_device


_adapter_arguments = {}
_device_arguments = {}
_shared_device_promise = None


def configure_shared_device(
    *,
    # Adapter arguments
    feature_level: str | None = None,
    power_preference: enums.PowerPreferenceEnum | None = None,
    force_fallback_adapter: bool | None = None,
    canvas: CanvasLike = None,
    # Device arguments
    label: str | None = None,
    required_features: Sequence[enums.FeatureNameEnum] = (),
    required_limits: dict[str, int | None] | None = None,
    # default_queue: structs.QueueDescriptorStruct | None = None,
):
    if _shared_device_promise is not None:
        raise RuntimeError(
            "configure_shared_device() is called but the shared device is already created."
        )

    # We want to do a good job validating inputs here, because the global device
    # can be requested from multiple places, and if an invalid arg is set from
    # one of these, and we use it to request the adapter, it's hard to establish
    # where the invalid arg was set from.

    if feature_level is not None:
        if not isinstance(feature_level, str):
            raise TypeError(
                f"configure_shared_device(..., feature_level) must be a str, but got {feature_level.__class__.__name__}."
            )
        if feature_level not in ("core", "compatibility"):
            raise ValueError(
                f"configure_shared_device(..., feature_level) must be a one of {'core', 'compatibility'}, but got {feature_level!r}."
            )
        cur_feature_level = adapter_arguments.get("feature_level", None)
        if cur_feature_level is not None and cur_feature_level != feature_level:
            logger.warning(
                f"configure_shared_device(..., feature_level) overrides earlier set {cur_feature_level!r} with {feature_level!r}."
            )

    if power_preference is not None:
        if not isinstance(power_preference, str):
            raise TypeError(
                f"configure_shared_device(..., power_preference) must be a str, but got {power_preference.__class__.__name__}."
            )
        if power_preference not in enums.PowerPreferenceEnum:
            raise ValueError(
                f"configure_shared_device(..., power_preference) must be a one of {enums.PowerPreferenceEnum}, but got {power_preference!r}."
            )
        cur_power_preference = adapter_arguments.get("power_preference", None)
        if (
            cur_power_preference is not None
            and cur_power_preference != power_preference
        ):
            logger.warning(
                f"configure_shared_device(..., power_preference) overrides earlier set {cur_power_preference!r} with {power_preference!r}."
            )


def request_shared_device(**kwargs):
    global _shared_device_promise

    if kwargs:
        configure_shared_device(**kwargs)

    if _shared_device_promise is None:
        promise1 = wgpu.gpu.request_adapter_async(**_adapter_arguments)

        def on_adapter(adapter: wgpu.GPUAdapter):
            return adapter.request_device_async(**_device_arguments)

        _shared_device_promise = promise1.then(
            on_adapter, title="request_shared_device"
        )

    return _shared_device_promise
