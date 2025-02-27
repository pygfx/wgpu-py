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

# run shader
out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={2: (np.prod((m, n)), "f")},
    shader=open(f"./matmul_simple.wgsl").read(),
    n=(n, m, 1),  # n cols across "x dimension", m rows across "y dimension"
)

# get output
C = np.frombuffer(out[2], dtype=np.float32).reshape((m, n))

# check that results are the same as numpy, we can expect 7 decimal precision
print(f"np.allclose():\n {np.allclose(A @ B, C)}\n")
print(f"AB - C:\n{A @ B - C}\n")
diff_norms = np.linalg.norm(A @ B - C, ord="fro") / np.linalg.norm(A @ B, ord="fro")
print(f"||AB - C||_F - ||AB||_F:\n{diff_norms}\n")
print(f"C:\n{C}")
