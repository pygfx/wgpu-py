"""Global configuration for pytest"""

import os

import numpy as np
import pytest


# Allow running the tests against a specific backend, e.g. WGPUPY_BACKEND=dawn.
# The backend must be loaded before any test module imports another backend.
if os.getenv("WGPUPY_BACKEND", "").strip():
    import wgpu.backends.auto  # noqa: F401


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-screenshots",
        action="store_true",
        dest="regenerate_screenshots",
        default=False,
    )


@pytest.fixture(autouse=True)
def predictable_random_numbers():
    """
    Called at start of each test, guarantees that calls to random produce the same output over subsequent tests runs,
    see https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.seed.html
    """
    np.random.seed(0)


@pytest.fixture(autouse=True, scope="session")
def numerical_exceptions():
    """
    Ensure any numerical errors raise a warning in our test suite
    The point is that we enforce such cases to be handled explicitly in our code
    Preferably using local `with np.errstate(...)` constructs
    """
    np.seterr(all="raise")
