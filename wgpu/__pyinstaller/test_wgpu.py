def test_pyi_wgpu(pyi_builder):
    pyi_builder.test_source(
        """
        import wgpu.resources
        import wgpu.backend.rs
    """
    )
