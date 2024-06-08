_default_device = None


def get_default_device(required_features=[], required_limits={}):
    """Get a wgpu device object. If this succeeds, it's likely that
    the WGPU lib is usable on this system. If not, this call will
    probably exit (Rust panic). When called multiple times,
    returns the same global device object (useful for e.g. unit tests).
    Parameters:
        required_features (list of str|`wgpu.FeatureName`): the features (extensions) that you need. Default [].
        required_limits (dict): the various limits that you need. Default {}.
    """
    global _default_device

    if _default_device is None:
        import wgpu.backends.auto  # noqa

        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        _default_device = adapter.request_device(
            required_features=required_features, required_limits=required_limits
        )
    return _default_device
