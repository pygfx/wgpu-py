import os
import tempfile
import subprocess

def glsl2spirv(glsl):
    filename1 = os.path.join(tempfile.gettempdir(), "x.vert")
    filename2 = os.path.join(tempfile.gettempdir(), "x.vert.spv")
    with open(filename1, "wb") as f:
        f.write(glsl.encode())

    try:
        stdout = subprocess.check_output(["glslangvalidator", "-V", filename1, "-o", filename2], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        e = "Could not compile glsl to Spir-V:\n" + err.output.decode()
        raise Exception(e)

    try:
        stdout = subprocess.check_output(["spirv-dis", filename2], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        e = "Could not disassemle Spir-V:\n" + err.output.decode()
        raise Exception(e)
    else:
        return stdout.decode()

## Naked function

print(glsl2spirv("""
#version 450

void main()
{
}
"""))


## One in, one out

print(glsl2spirv("""
#version 450

//layout (location = 12) in vec3 aPos; // the position variable has attribute position 0

layout(location = 13) out vec4 vertexColor; // specify a color output to the fragment shader

void main()
{
    vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
    //vertexColor = vec4(aPos, 1.0);
}
"""))


## Builtin out vars

print(glsl2spirv("""
#version 450

void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""))


## Uniforms

print(glsl2spirv("""
#version 450

layout(binding = 0) uniform meeeeh {
    vec3 uColor;
} uniform_object;

void main()
{
}
"""))


## Constant

print(glsl2spirv("""
#version 450

vec3 uColor = vec3(1.0, 0.0, 0.0);

void main()
{
}
"""))


## Vector composite

print(glsl2spirv("""
#version 450

void main()
    {
    int index = 0;
    vec2 positions[3] = vec2[3]( vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5) );
    vec2 x = positions[index];
}
"""))
