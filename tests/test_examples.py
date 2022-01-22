"""
Test the force offscreen auto gui mechanism.
"""

import os
from pathlib import Path
import importlib

import numpy as np

import pytest
import wgpu.backends.rs  # noqa
from testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


examples_dir = Path(__file__).parent.parent / "examples"
examples_to_test = []
for example_path in examples_dir.glob("*.py"):
    example_code = example_path.read_text()
    if "wgpu.gui.auto" in example_code and "canvas = WgpuCanvas(" in example_code:
        examples_to_test.append(example_path.stem)


@pytest.fixture(autouse=True, scope="module")
def force_offscreen():
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


@pytest.mark.parametrize("module", examples_to_test)
def test_examples(module):
    """Run every example that supports testing via the auto gui mechanism."""
    example = importlib.import_module(f"examples.{module}")

    img = example.canvas.draw()

    # assert something was rendered
    assert img is not None
    # assert the image is not black or of nil size
    assert img.size > 0
    assert not np.all(img == 0)
