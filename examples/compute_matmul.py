"""
Simple compute example that performs basic matrix multiplication.

Uses linear arrays in storage buffers to represent matrices of arbitrary size since
the wgsl standard library only supports matrix multiplication upto 4x4 matrices.
"""

import numpy as np
from wgpu.utils.compute import compute_with_buffers

# define matrix shapes
m, k, n = 6, 7, 8

A_shape = (m, k)
B_shape = (k, n)

# create random matrices
A = (
    np.random.rand(A_shape[0] * A_shape[1])
    .astype(np.float32)
    .reshape(A_shape, order="C")
)
B = (
    np.random.rand(B_shape[0] * B_shape[1])
    .astype(np.float32)
    .reshape(B_shape, order="C")
)

# define bindings
bindings = {
    0: A,
    1: B,
    3: np.array(A_shape, dtype=np.uint32),
    4: np.array(B_shape, dtype=np.uint32),
}

shader_src = """
@group(0) @binding(0)
var<storage, read> A: array<f32>;
@group(0) @binding(1)
var<storage, read> B: array<f32>;
@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<storage, read> A_shape: array<u32>;
@group(0) @binding(4)
var<storage, read> B_shape: array<u32>;


fn get_1d_index(row_ix: u32, col_ix: u32, n_cols: u32) -> u32 {
    // get the 1D index in the array which corresponds
    // to the passed row and column index
    return row_ix * n_cols + col_ix;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // naive matrix multiplication
    // A ∈ R^(m x k), B ∈ R^(k x n), AB = C ∈ R^(m x n)
    // gid.y is "m index", gid.x is "n index"

    // we make these varibles because we cannot pass individual array elements to a function
    // i.e. get_1d_index(A_shape[0]) is not possible
    let m: u32 = A_shape[0];
    let k: u32 = A_shape[1];
    let n: u32 = B_shape[1];

    // computes one element of C using A at row gid.y and B at column gid.x
    var sum: f32 = 0.0;

    // computes one element of C using A at row gid.y and B at column gid.x
    for (var i: u32 = 0; i < k; i++) {
        // dot product of A at row = gid.y, col = i and B at row = i, col = gid.x
        // A col max index is k - 1, B row max index is k - 1
        sum = sum + A[get_1d_index(gid.y, i, k)] * B[get_1d_index(i, gid.x, n)];
    }

    // set element of C
    C[get_1d_index(gid.y, gid.x, n)] = sum;

    return;
}
"""

# run shader
out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={2: (np.prod((m, n)), "f")},
    shader=shader_src,
    n=(n, m, 1),  # n cols across "x dimension", m rows across "y dimension"
)

# get output
C = np.frombuffer(out[2], dtype=np.float32).reshape((m, n))

# check that results are the same as numpy, we can expect 7 decimal precision
all_close = np.allclose(A @ B, C)
assert all_close
print(f"np.allclose():\n {all_close}\n")
print(f"AB - C:\n{A @ B - C}\n")
diff_norms = np.linalg.norm(A @ B - C, ord="fro") / np.linalg.norm(A @ B, ord="fro")
print(f"||AB - C||_F - ||AB||_F:\n{diff_norms}\n")
print(f"C:\n{C}")
