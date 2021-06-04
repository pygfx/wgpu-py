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


### [v0.6] - (upcoming)

No changes yet.


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

This release uses a new version of wgpu-native which has changed quite a bit iternally. There
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
