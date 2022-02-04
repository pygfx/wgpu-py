"""
Test that the examples run without error.
"""

import os
from pathlib import Path
import importlib
from unittest.mock import patch

import pytest
from testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)

examples_dir = Path(__file__).parent.parent / "examples"

# find examples that contain the marker comment for inclusion in the test suite
MARKER_COMMENT = "# test_example = true"
examples_to_test = []
for example_path in examples_dir.glob("*.py"):
    example_code = example_path.read_text()
    if MARKER_COMMENT in example_code:
        examples_to_test.append(example_path.stem)


@pytest.fixture(autouse=True, scope="module")
def force_offscreen():
    """Force the offscreen canvas to be selected by the auto gui module."""
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


@pytest.fixture(autouse=True)
def mock_time():
    """Some examples use time to animate. Fix the return value
    for repeatable output."""
    with patch("time.time") as time_mock:
        time_mock.return_value = 1.23456
        yield


@pytest.mark.parametrize("module", examples_to_test)
def test_examples(module):
    """Run every example that supports testing via the auto gui mechanism."""
    example = importlib.import_module(f"examples.{module}")

    # render
    img = example.canvas.draw()

    # assert something was rendered
    assert img is not None and img.size > 0
