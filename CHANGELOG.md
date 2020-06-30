# The changelog

WGPU and WebGPU are still changing fast, and with that we do to. We don
not yet attempt to make things backwards compatible. Instead we try to
be precise about tracking changes to the public API.


### v0.3.0

With this update we're using a later release of wgpu-native, and follow changes
is the WebGPU spec. Further, we've removed the need for ctypes to communicate
data arrays. Instead, wgpu-py can consume any object that supports the buffer
protocol, and it returns `memoryview` objects.

Additions and improvements:

* Targets wgpu-native v.0.5.2. The first release build from the wgpu-native repo itself.
* The texture object has more properties to query the parameters that it was created with.
* The texture view object has a `texture` property.
* The buffer object has a property `map_mode`.
* The render and compute pipeline objects have a property `layout` and a method `get_bind_group_layout()`.
* The shader object got a `compilation_info` method, but this does not do anything yet.
* The `create_shader_module()` has a `source_map` attribute, but this is yet unused.
* Log messages from wgpu-native (Rust) are now injected into Python's logger.
* The `queue` object got two new methods `write_buffer` and `write_texture`.

API changes:

* The buffer no longer has a `mapped` attribute.
* The buffer `map_read` and `map_write` methods have been replaced with `map`,
  where the first argument is e.g. `wgpu.MapMode.READ`. This method returns
  a `memoryview` object that can be read from or written to.
* The device `create_buffer_mapped` method now returns a tuple (buffer, mapped_mem).
* The device `create_buffer_with_data` is added as a convenience function.
* The `array_layer` in copy operations involving a texture is removed.
* The `utils.compute_with_buffers` function now accepts *any* data dat supports
  the buffer protocol (not just ctypes arrays). The outputs are `memoryview` objects,
  which shape and format can be specified. When a ctypes array type is specified,
  the output will be an instance of that type. This means that these changes are
  fully backwards compatible.


### v0.2.0

Additions and improvements:

* Targets wgpu-native v0.5.1. This is the last release when wgpu-native was still part of wgpu-core.
* The canvase now has a `request_draw` method.
* More and better docs.
* The canvas can be passed to `request_adapter` so that the created surface
  can be selected on it.
* Support for debug markers.

API changes:

* `bindings` in bind groups and bind group layouts are now called `entries`.
* There is no more generic storage texture, only a readonly and a writeonly one.
* `set_index_buffer` and `set_vertex_buffer` got a `size` argument.
* `row_pitch` and `image_height` in copy operations involving a texture
  are renamed to `bytes_per_row` and `rows_per_image`.
* Rendering is now done under the swap_chain's context: `with swap_chain as current_texture_view`


### v0.1.6

This release is the first moderately mature version of wgpu-py.
