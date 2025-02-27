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
