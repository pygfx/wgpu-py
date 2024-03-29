"""
A collection of tests related to non-finite values in shaders, like nan and inf.

See:
* https://en.wikipedia.org/wiki/NaN
* https://github.com/gpuweb/gpuweb/pull/2311#issuecomment-1973533433

"""

import ctypes

import numpy as np

from wgpu.utils.compute import compute_with_buffers
from pytest import skip
from testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_finite_using_nequal():
    # Just to demonstrate that this does not work.
    # The compiler filters optimizes away the check.

    shader = """
        @group(0)
        @binding(0)
        var<storage,read> values: array<f32>;

        fn is_nan(v:f32) -> bool {
            return v != v;
        }

        fn is_inf(v:f32) -> bool {
            return v != 0.0 && v * 2.0 == v;
        }

        fn is_finite(v:f32) -> bool {
            return v == v && v * 2.0 != v;
        }

        fn to_real(v:f32) -> f32 {
            return select(0.0, v, is_finite(v));
        }

    """

    detect_finites("nequal", shader, False, False)


def test_finite_using_min_max():
    # This obfuscates the check for equality enough for the compiler
    # not to optimize it away.
    #
    # However, if fastmath is enabled, depending on the hardare/compiler,
    # the loaded value may not actually be a nan/inf anymore.

    shader = """
        @group(0)
        @binding(0)
        var<storage,read> values: array<f32>;

        fn is_nan(v:f32) -> bool {
            return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0;
        }

        fn is_inf(v:f32) -> bool {
            return v != 0.0 && v * 2.0 == v;
        }

        fn is_finite(v:f32) -> bool {
            return !is_nan(v) && !is_inf(v);
        }

        fn to_real(v:f32) -> f32  {
            return select(0.0, v, is_finite(v));
        }

    """

    detect_finites("min-max", shader, True, True)


def test_finite_using_uint():
    # This is the most reliable approach.

    shader = """
        @group(0)
        @binding(0)
        var<storage,read> values: array<u32>;

        fn is_nan(v:u32) -> bool {
            let mask = 0x7F800000u;
            let v_is_pos_inf = v == 0x7F800000u;
            let v_is_neg_inf = v == 0xFF800000u;
            let v_is_finite = (v & mask) != mask;
            return !v_is_finite & !(v_is_pos_inf | v_is_neg_inf);
        }

        fn is_inf(v:u32) -> bool {
            let v_is_pos_inf = v == 0x7F800000u;
            let v_is_neg_inf = v == 0xFF800000u;
            return v_is_pos_inf | v_is_neg_inf;
        }

        fn is_finite(v:u32) -> bool {
            return (v & 0x7F800000u) != 0x7F800000u;
        }

        fn to_real(v:u32) -> f32 {
            return select(0.0, bitcast<f32>(v), is_finite(v));
        }
    """

    detect_finites("uint", shader, True, True)


def detect_finites(title, shader, expect_detection_nan, expect_detection_inf):

    base_shader = """

        @group(0)
        @binding(1)
        var<storage,read_write> result_nan: array<i32>;

        @group(0)
        @binding(2)
        var<storage,read_write> result_inf: array<i32>;

        @group(0)
        @binding(3)
        var<storage,read_write> result_finite: array<i32>;

        @group(0)
        @binding(4)
        var<storage,read_write> result_real: array<f32>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i = i32(index.x);
            let value = values[i];

            result_nan[i] = i32(is_nan(value));
            result_inf[i] = i32(is_inf(value));
            result_finite[i] = i32(is_finite(value));
            result_real[i] = to_real(value);

        }

    """

    # Create data in blocks of 10: zeros, nans, infs, random reals
    parts = [
        [0.0] * 10,
        [
            float("nan"),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            float("-inf"),
            float("inf"),
            -np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ],
        np.random.uniform(-1e9, 1e9, 10),
    ]
    values = np.concatenate(parts, dtype=np.float32)

    # Check length
    assert values.shape == (40,)
    n = len(values)

    # Create reference bool arrays
    is_nan_ref = np.zeros((n,), bool)
    is_nan_ref[10:20] = True
    is_inf_ref = np.zeros((n,), bool)
    is_inf_ref[20:30] = True
    is_finite_ref = np.ones((n,), bool)
    is_finite_ref[10:30] = False

    # Get reference real array
    real_ref = values.copy()
    real_ref[~is_finite_ref] = 0

    # Compute!
    out = compute_with_buffers(
        {0: (ctypes.c_float * n)(*values)},
        {
            1: n * ctypes.c_int32,
            2: n * ctypes.c_int32,
            3: n * ctypes.c_int32,
            4: n * ctypes.c_float,
        },
        shader + base_shader,
    )
    is_nan = out[1]
    is_inf = out[2]
    is_finite = out[3]
    real = out[4]

    # Check that numpy detects ok
    assert np.all(np.isnan(values) == is_nan_ref)
    assert np.all(np.isinf(values) == is_inf_ref)
    assert np.all(np.isfinite(values) == is_finite_ref)

    # Check that our shader does too
    detected_nan = bool(np.all(is_nan == is_nan_ref))
    detected_inf = bool(np.all(is_inf == is_inf_ref))
    detected_finite = bool(np.all(is_finite == is_finite_ref))
    good_reals = bool(np.all(real == real_ref))

    # Print, for when run as a script
    checkmark = lambda x: "x✓"[x]  # noqa
    print(
        f"{title:>10}:   {checkmark(detected_nan)} is_nan   {checkmark(detected_inf)} is_inf   {checkmark(detected_finite)} is_finite   {checkmark(good_reals)} good_reals"
    )

    # Test
    if expect_detection_nan:
        assert detected_nan
    if expect_detection_inf:
        assert detected_inf
    if expect_detection_nan and expect_detection_inf:
        assert detected_finite
        assert good_reals


if __name__ == "__main__":

    test_finite_using_nequal()
    test_finite_using_min_max()
    test_finite_using_uint()
