import wgpu.utils

from testutils import run_tests
from pytest import raises

dedent = lambda s: s.replace("\n        ", "\n").strip()


def test_parse_shader_error1(caplog):
    # test1: invalid attribute access
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec4<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
            var out: VertexOutput;
            out.invalid_attr = vec4<f32>(0.0, 0.0, 1.0);
            return out;
        }
    """

    expected = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader '' parsing error: invalid field accessor `invalid_attr`
          ┌─ wgsl:9:9
          │
        9 │     out.invalid_attr = vec4<f32>(0.0, 0.0, 1.0);
          │         ^^^^^^^^^^^^ invalid accessor
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    error = error.rstrip("\n")
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error2(caplog):
    # test2: grammar error, expected ',', not ';'
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>;
            @builtin(position) position: vec4<f32>,
        };
    """

    expected = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader '' parsing error: expected `,`, found ";"
          ┌─ wgsl:2:38
          │
        2 │     @location(0) texcoord : vec2<f32>;
          │                                      ^ expected `,`
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    error = error.rstrip("\n")
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error3(caplog):
    # test3: grammar error, contains '\t' and (tab),  unknown scalar type: 'f3'
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec4<f3>,
        };
    """

    expected = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader '' parsing error: unknown type: `f3`
          ┌─ wgsl:3:39
          │
        3 │     @builtin(position) position: vec4<f3>,
          │                                       ^^ unknown type
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    error = error.rstrip("\n")
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error4(caplog):
    # test4: no line info available - hopefully Naga produces better error messages soon?
    device = wgpu.utils.get_default_device()

    code = """
        fn foobar() {
            let m = mat2x2<f32>(0.0, 0.0, 0.0, 0.);
            let scales = m[4];
        }
    """

    expected = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader '' parsing error: Index 4 is out of bounds for expression [10]
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    error = error.rstrip("\n") # seems to have tailing newlines sometimes?
    assert error == expected, f"Expected:\n\n{expected}"


def test_validate_shader_error1(caplog):
    # test1: Validation error, mat4x4 * vec3
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec3<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
            var out: VertexOutput;
            var matrics: mat4x4<f32>;
            out.position = matrics * out.position;
            return out;
        }
    """

    expected1 = """Left: Load { pointer: [2] } of type Matrix { columns: Quad, rows: Quad, scalar: Scalar { kind: Float, width: 4 } }"""
    expected2 = """Right: Load { pointer: [5] } of type Vector { size: Tri, scalar: Scalar { kind: Float, width: 4 } }"""
    expected3 = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader validation error: Entry point vs_main at Vertex is invalid
           ┌─ :10:20
           │
        10 │     out.position = matrics * out.position;
           │                    ^^^^^^^^^^^^^^^^^^^^^^ naga::ir::Expression [7]
           │
           = Expression [7] is invalid
           = Operation Multiply can't work with [4] (of type Matrix { columns: Quad, rows: Quad, scalar: Scalar { kind: Float, width: 4 } }) and [6] (of type Vector { size: Tri, scalar: Scalar { kind: Float, width: 4 } })


              Expression [7] is invalid
                Operation Multiply can't work with [4] (of type Matrix { columns: Quad, rows: Quad, scalar: Scalar { kind: Float, width: 4 } }) and [6] (of type Vector { size: Tri, scalar: Scalar { kind: Float, width: 4 } })
    """

    code = dedent(code)
    expected3 = dedent(expected3)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert caplog.records[1].msg == expected2
    assert err.value.message.strip() == expected3, f"Expected:\n\n{expected3}"


def test_validate_shader_error2(caplog):
    # test2: Validation error, multiple line error, return type mismatch
    device = wgpu.utils.get_default_device()

    code = """
        struct Varyings {
            @builtin(position) position : vec4<f32>,
            @location(0) uv : vec2<f32>,
        };

        @vertex
        fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
            if (in.uv.x > 0.5) {
                return vec3<f32>(1.0, 0.0, 1.0);
            } else {
                return vec3<f32>(0.0, 1.0, 1.0);
            }
        }
    """

    expected1 = """Returning Some(Handle([3])) where Some([0]) is expected"""
    expected2 = """
        Validation Error

        Caused by:
          In wgpuDeviceCreateShaderModule

        Shader validation error: Entry point fs_main at Vertex is invalid
          ┌─ :9:16
          │
        9 │         return vec3<f32>(1.0, 0.0, 1.0);
          │                ^^^^^^^^^^^^^^^^^^^^^^^^ naga::ir::Expression [8]
          │
          = The `return` expression Some([8]) does not match the declared return type Some([0])


              The `return` expression Some([8]) does not match the declared return type Some([0])
    """

    code = dedent(code)
    expected2 = dedent(expected2)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert err.value.message.strip() == expected2, f"Expected:\n\n{expected2}"


if __name__ == "__main__":
    run_tests(globals())
