def create_device():
    """ Create a wgpu device object. If this succeeds, it's likely that
    the WGPU lib is usable on this system. If not, this call will
    probably exit (Rust panic). returns None.
    """
    import wgpu.backend.rs

    wgpu.request_adapter(power_preference="high-performance").request_device()
