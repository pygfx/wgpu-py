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


### [v0.4] - future

* Maybe restore functionality for mapped buffers.


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
