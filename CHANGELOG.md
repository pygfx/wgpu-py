# Changelog / release notes

WGPU and WebGPU are still changing fast, and with that we do to. We dont
not yet attempt to make things backwards compatible. Instead we try to
be precise about tracking changes to the public API.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


Possible sections in each release:

* Added: for new features.
* Changed: for changes in existing functionality.
* Deprecated: for soon-to-be removed features.
* Removed: for now removed features.
* Fixed: for any bug fixes.
* Security: in case of vulnerabilities.


### [v0.7.4] - 04-02-2022

Fixed:

* Position of glfw pointer events on MacOS.


### [v0.7.3] - 25-01-2022

Added:

* Expanded the `auto` gui backend, which can now also select qt framework if available.
* The qt gui backend (like the glfw gui backend) supports user events in the same manner as
  the jupyter gui backend.
* Expanded the `auto` gui backend to also support an offscreen canvas intended for automated tests.

Fixed:

* Size of glfw windows on MacOS.


### [v0.7.2] - 24-12-2021

Fixed:

* Exceptions in user-interaction callbacks don't break the glfw loop anymore.
* Pointer events in glfw have the correct key modifiers now.


### [v0.7.1] - 22-12-2021

Added:

* #224 - Added `add_event_handler` and `remove_event_handler` to GLFW and Jupyter GUI canvases.


### [v0.7.0] - 21-12-2021

Changed:

* Now targeting wgpu-native v0.11.0.1, containing many upstream fixes and improvements.
* The `[[block]]` syntax in shaders has been dropped.
* Renamed `ProgrammableStage.clamp_depth` -> `unclipped_depth`.


### [v0.6.0] - 16-12-2021

Added:

* Official support for Windows 3.10.
* The `max_fps` argument can be provided to a canvas.
* The glfw gui backend supports user events in the same manner as the jupyter gui backend,
  using the [jupyter_rfb event specification](https://jupyter-rfb.readthedocs.io/en/latest/events.html).
* Introduce the `auto` gui backend, which selects either glfw or jupyter.

Fixed:

* The wx gui backend is now fully functional.

Changed:

* The qt and wx gui backend now contain `WgpuCanvas` for a toplevel window,
  and `WgpuWidget` for an embeddable widget.
* All gui backends (can) now limit the FPS.
* No changes to the wgpu API.


### [v0.5.9] - 11-10-2021

Fixed:

* Include the correct binaries in macOS arm64 wheels
* Options for arch argument of download-wgpu-native.py script


### [v0.5.8] - 09-10-2021

Added:

* Experimental support for macos_arm64 (M1).

Changed:

* The Qt examples use PySide6 instead of PyQt5.


### [v0.5.7] - 07-10-2021

Changed:

* Update to the latest wgpu-native (including latest Naga).
* The power-preference is actually taken into account.
* The adapter actually reports its limits.
* The limits in `request_device` are actually used.
* The `Adapter.is_software` property is renamed to `Adapter.is_fallback_adapter`.


### [v0.5.6] - 30-08-2021

Added:

* An offscreen canvas to take snapshots without needing a window.

Changed:

* On Windows, the Vulkan backend is now forced unless `WGPU_BACKEND_TYPE` is set.

Fixed:

* Better support for multiple canvases by fixing a specific Qt issue.
* Fixed that canvas was not passed to low level function of `request_adapter`.
* Support calling `get_current_texture()` multiple times during a draw.


### [v0.5.5] - 09-08-2021

Added:

* The wgpu backend can be forced using the `WGPU_BACKEND_TYPE` env variable.
  Values can be e.g. "D3D12", "Metal", "Vulkan".
* Initial support for off-screen canvases.
* Adds `adapter.is_software` property.

Changed:

* The `GPUPresentationContext` class has been renamed to `GPUCanvasContext`.
* The functionality of the swap-chain has moved to the `GPUCanvasContext`.
* The now removed `GPUSwapChain` was used as a context manager. Instead,
  the frame is presented (ala GL swapbuffers) automatically at the end of a draw.
* The `canvas.configure_swap_chain()` method has been removed. Instead,
  `canvas.get_context()` should be used, to obtain a present/canvas context.
* The `adapter.request_device()` method has its arguments `non_guaranteed_features`
  and `non_guaranteed_limits` replaced with `required_features` and `required_limits`.
* The enum field `StoreOp.clear` is now `StoreOp.discard`.
* The flag field `TextureUsage.SAMPLED ` is now `TextureUsage.TEXTURE_BINDING `.
* The flag field `TextureUsage.STORAGE ` is now `TextureUsage.STORAGE_BINDING `.
* The enum `InputStepMode` is now `VertexStepMode`.
* WGSL: `arrays` must be declared as `var` (not `let`) in order to allow dynamic indexing.
* WGSL: storage classes are written differently.


### [v0.5.4] - 11-06-2021

Changed:

* The backend selection is automatic by default. To force a backend, the `WGPU_BACKEND_TYPE` evironment variable can be set to e.g. "Vulkan". It could be good to do this on Windows to prevent selection of DX12 for now.


### [v0.5.3] - 04-06-2021

Added:

* `adapter.properties` now has actual values, allowing inspeciton of the selected
    GPU and backend.
* Added back support for filtering float32 textures by enabling a certain wgpu feature
  by default.

Fixed:

* An error in the docs of `create_render_pipeline`.
* Vulkan backend is now forced to prevent DX12 being select and causing probems
  because it's less mature.


### [v0.5.2] - 23-05-2021

This release uses a new version of wgpu-native which has changed quite a bit internally. There
is more validation (thus more restrictions). There are only a few changes to the API.
However, one big change is that shaders can now be provided as both SpirV and WGSL. Due to
the strict validation, most shaders compiled by PyShader are not usable anymore. We
recommend using WGSL instead.

Added:

* Added `GPUAdaper.properties` (the amount of information it contains will increase in the future).
* Added proper support for WGSL.

Changed:

* Renamed `renderpass.set_blend_color` -> `set_blend_constant`.
* Stricter validation of SpirV shaders.
* Float32 texture formats must now use a non-filtering sampler and texture-sample-type.
* Integer texture formats can no longer use a texture (use `textureLoad` instead).
* ... and more tighter restrictions.

Removed:

* The API concerning debug markers and groups is temporarily removed.
* Adapter and device features is temporarily removed.
* Adapter and device limits is temporarily removed.


### [v0.4] - 21-05-2021

This release represents about half a year of progress on the WebGPU API, so the API
has changed quite a bit. The wgpu-py API more closely reflects the webgpu API - wgpu-native does
not affect the API except for a few additional features.

Added:

* Added `GPUQueue.read_buffer` as extra API (next to `write_buffer` which is original WebGPU API).
* Added `GPUQueue.read_texture` as extra API.

y
Removed:

* Removed `GPUBuffer.read_data()`. Use `device.queue.read_buffer()` instead. Note that `usage` `MAP_READ` should be replaced with `COPY_SRC`.
* Removed `GPUBuffer.write_data()`. Use `device.queue.write_buffer()` instead. Note that `usage` `MAP_WRITE` should be replaced with `COPY_DST`.

Changed:

* `GPUCanvasContext.get_swap_chain_preferred_format()`: now takes an `adapter` instead of a `device`.
* `GPUAdapter.extensions`: is now called `features`.
* `GPUAdapter.request_device()`: the `extensions` and `limit` args are now `non_guaranteed_features` and `non_guaranteed_limits`.
* `GPUDevice.default_queue`: is now called `queue`.
* `GPUDevice.create_compute_pipeline()`: the `compute_stage` arg is now called `compute`.
* `GPUDevice.create_bind_group_layout()` has changed the required structure of the layout enty dicts.
* `GPUDevice.create_render_pipeline()` has changed *a lot* in terms of shape of input dicts. See new docs.
* `GPUTexture.create_view()`: args `mip_level_count` and `array_layer_count` are default `None` instead of `0`.
* `GPUCommandEncoder.begin_render_pass()`: the `color_attachments` and `depth_stencil_attachment` arguments have their `attachment` field renamed to `view`.
* `GPURenderEncoderBase.set_index_buffer()` has an extra argument (after the buffer) to specify the format. The index format is no longer specified in `device.create_render_pipeline()`.
* Flag `TextureUsage` has field OUTPUT_ATTACHMENT renamed to RENDER_ATTACHMENT.
* Enum `BindingType` is split up in different enums for buffer, sampler, sampled texture and storage texture.
* Enum `BlendFactor` has some of its field names changed.
* Enum `VertexFormat` has its field names changed, e.g. ushort2 -> uint16x2.
* The API is more restrictive in the use of buffer/texture usage combinations.
* The API is more restrictive in formats for storage buffers/textures.
* When copying from/to textures, the `bytes_per_row` must now be a multiple of 256.


### [v0.3.0] - 2020-07-05

With this update we're using a later release of wgpu-native, and follow changes
is the WebGPU spec. Further, we've removed the need for ctypes to communicate
data arrays. Instead, wgpu-py can consume any object that supports the buffer
protocol, and it returns `memoryview` objects.

Added:

* The texture object has more properties to query the parameters that it was created with.
* The texture view object has a `texture` property.
* The render and compute pipeline objects have a property `layout` and a method `get_bind_group_layout()`.
* The shader object got a `compilation_info` method, but this does not do anything yet.
* The `create_shader_module()` has a `source_map` attribute, but this is yet unused.
* Log messages from wgpu-native (Rust) are now injected into Python's logger.
* The `queue` object got two new methods `write_buffer` and `write_texture`.
* The buffer has `read_data()` and `write_data()` methods. Note: the latter may be removed later.
* The device `create_buffer_with_data` is added as a convenience function. This will likely stay.

Changed:

* Targets wgpu-native v.0.5.2. The first release build from the wgpu-native repo itself.
* The `array_layer` in copy operations involving a texture is removed.
* The `utils.compute_with_buffers` function now accepts *any* data dat supports
  the buffer protocol (not just ctypes arrays). The outputs are `memoryview` objects,
  which shape and format can be specified. When a ctypes array type is specified,
  the output will be an instance of that type. This means that these changes are
  fully backwards compatible.

Removed:

* The buffer (for now) no longer exposes a data mapping API. Instead use `read_data()` and `write_data()`.
* The device `create_buffer_mapped` method is similarly removed. Use `create_buffer_with_data` instead.


### [v0.2.0] - 2020-04-16

Added:

* The canvase now has a `request_draw` method.
* More and better docs.
* The canvas can be passed to `request_adapter` so that the created surface
  can be selected on it.
  * Support for debug markers.

Changed:

* Targets wgpu-native v0.5.1. This is the last release when wgpu-native was still part of wgpu-core.
* The `bindings` in bind groups and bind group layouts are now called `entries`.
* There is no more generic storage texture, only a readonly and a writeonly one.
* The `set_index_buffer` and `set_vertex_buffer` methods got a `size` argument.
* The `row_pitch` and `image_height` args in copy operations involving a texture
  are renamed to `bytes_per_row` and `rows_per_image`.
* Rendering is now done under the swap_chain's context: `with swap_chain as current_texture_view`


### [v0.1.6] - 2020-04-01

This release is the first moderately mature version of wgpu-py.
