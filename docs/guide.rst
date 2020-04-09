-----
Guide
-----

Not a lot here yet. More will come over time.


A brief history of WGPU
-----------------------

For years, OpenGL has been the only cross-platform API to talk to the GPU.
But over time OpenGL has grown into an inconsistent and complex API ...

    *OpenGL is dying*
    --- Dzmitry Malyshau at `Fosdem 2020 <https://fosdem.org/2020/schedule/event/rust_webgpu/>`_

In recent years, modern API's have emerged that solve many of OpenGL's problems.
You may have heard of them: Vulkan, Metal, and DX12. These API's are
much closer to the hardware. Unfortunately, the huge amount of "knobs
to turn" makes them quite hard to work with for developers.

Therefore, people are working on a higher level API, that wraps Vulkan/Metal/DX12,
and uses the same principals, but is much easier to work with. This is the
`WebGPU spec <https://gpuweb.github.io/gpuweb/>`_. This is what future devs
will be using to write GPU code for the browser. And for desktop and mobile.

As WebGPU spec is being developed, a reference implementation is also
being build. It's written in Rust, and is likely going to power the
WebGPU implementation in Firefox.
This reference implementation, called `wgpu-native <https://github.com/gfx-rs/wgpu>`_,
also exposes a C-api, which means that it can be wrapped in Python. And this is what
wgpu-py does.

So in short, wgpu-py is a Python wrapper of wgpu-native, which is a wrapper
for Vulkan, Metal and DX12, which are low-level API's to talk to the GPU hardware.


Getting started with WGPU
-------------------------

For now, we'll direct you to some related tutorials:

* https://sotrh.github.io/learn-wgpu/
* https://rust-tutorials.github.io/learn-wgpu/


Examples
--------

Some examples with wgpu-py can be found here:

* https://github.com/almarklein/wgpu-py/tree/master/examples
