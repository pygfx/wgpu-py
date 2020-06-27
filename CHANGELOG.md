# The changelog

WGPU and WebGPU are still changing fast, and with that we do to. We don
not yet attempt to make things backwards compatible. Instead we try to
be precise about tracking changes to the public API.


### v0.3.0

Additions and improvements:

* Targets wgpu-native v.0.5.2. The first release build from the wgpu-native repo itself.
* Several GPU objects (most notably the texture) have more properties to query
  the parameters that they were created with.
* The shader object got a `compilation_info` method, but it does not do anything yet.

API changes:

* The `offset` and `size` arguments in `set_index_buffer` and `set_vertex_buffer`
  have become optional.
* The buffer's `map_read` and `map_write` methods have been replaced with `map`,
  where the first argument is e.g. `wgpu.MapMode.READ`.
* The `array_layer` in copy operations involving a texture is removed.


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
* `set_index_buffer` and `set_vertex_buffer` got a mandatory `size` argument.
* `row_pitch` and `image_height` in copy operations involving a texture
  are renamed to `bytes_per_row` and `rows_per_image`.
* Rendering is now done under the swap_chain's context: `with swap_chain as current_texture_view`


### v0.1.6

This release is the first moderately mature version of wgpu-py.
