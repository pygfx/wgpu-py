# Using the WGPU API

## Package structure

* _gen.py parses wgpu.h and generates the wgpu.py, which defines the abstract API.
* _gen.py also generates ctypes and ffi wrapper to wgpu C-API (written in Rust).
* wgpu_gl.py is handwritten (incomplete) gl implementation.
* wgpu_ctypes.py is an auto-generated wrapper for the Rust wgpu lib, not working.
* wgpu_ffi.py is an ffi-based auto-generated wrapper for the Rust wgpu lib, working!
* test_triangle.py is a test script to run.


## Status

Working:

* Can parse wgpu.h (well, 95% of it) and generate an abstract Python class.
* Implemented a triangle example based on this nice API.
* Have implemented part of the API using OpenGL, to get the triangle example running.
* Can build the wgpu_native.dll using Rust on Windows.
* GL backend works with GLFW (using glsl).
* FFI backend works with Qt (using spirv from the wgpu example).

Not working / roadblocks:

* Can build the wgpu C examples, but (at least the triangle one) is not working (it aborts).
  Don't really need these, but it would be a confirmation that the wgpu actually works on Windows.
* The wgpu_ctypes.py does not work: Rust keeps saying that there are no backends. There must be something
  wrong with the struct that I send ..
* GLWF works with the GL backend, but for the ffi backend I need the window handle, and
  this seems to be hard to get from glfw.
* Qt works with the ffi backend (we can get the window handle), but for the gl backend
  I have not gotten swapbuffers/context right.
