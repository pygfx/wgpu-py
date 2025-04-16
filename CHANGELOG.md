# Changelog / release notes

WebGPU and wgpu-native are still changing fast, and with that we do to. We do
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


### [v0.22.0] - 16-04-2025

Added:

* Implement wgpu extension writeTimestamp() by @fyellin in https://github.com/pygfx/wgpu-py/pull/690
* Implement proper support for alt texture formats for offscreen use by @almarklein in https://github.com/pygfx/wgpu-py/pull/701

Changed:

* Make ``adapter.info`` return GPUAdapterInfo by @fyellin in https://github.com/pygfx/wgpu-py/pull/699
* Give AdapterInfo a repr by @almarklein in https://github.com/pygfx/wgpu-py/pull/707

Fixed:

* Fix constructing QApplication for PyQt6 by @almarklein in https://github.com/pygfx/wgpu-py/pull/694
* Improve reporting when context.get_current_texture() fails by @almarklein in https://github.com/pygfx/wgpu-py/pull/702
* Improve get_current_texture for window resizing by @almarklein in https://github.com/pygfx/wgpu-py/pull/705
* Imgui get up-to-date framebuffer size by @almarklein in https://github.com/pygfx/wgpu-py/pull/706

Other:

* Add return type hints by @Vipitis in https://github.com/pygfx/wgpu-py/pull/697
* Verify that override variables can be used for workgroup sizes and array lengths by @fyellin in https://github.com/pygfx/wgpu-py/pull/692


### [v0.21.1] - 20-03-2025

This release updates to wgpu-native v24.0.3.1, which includes several upstream fixes.

Added:

* Add `multi_draw_indexed_indirect_count` and `multi_draw_indirect_count` by @fyellin in https://github.com/pygfx/wgpu-py/pull/687


### [v0.21.0] - 17-03-2025

This releases updates to using wgpu-native v24.0.0.2, and updates the API to the latest WebGPU spec (IDL).

Added:

* New `canvas_context.get_configuration()` method.
* New `device.adapter_info` property.
* New `usage` arg for `texture.create_view()`.
* New `feaure_level` arg for `request_adapter()`. More a diagnostic tool, not for most end-users.


### [v0.20.1] - 14-03-2025

Other:
* Add release notes.


### [v0.20.0] - 14-03-2025

Added:

* Add Imgui based Stats component by @panxinmiao in https://github.com/pygfx/wgpu-py/pull/670 and https://github.com/pygfx/wgpu-py/pull/674
* Can pass constants to compute helper by @kushalkolar in https://github.com/pygfx/wgpu-py/pull/676
* Add simple matmul example by @kushalkolar in https://github.com/pygfx/wgpu-py/pull/677

Fixed:

* Fix imgui scroll event by @panxinmiao in https://github.com/pygfx/wgpu-py/pull/678

Changed:

* Improve error handling by @almarklein in https://github.com/pygfx/wgpu-py/pull/667
* Bring sleep back to WgpuAwaitable. Slowly back off on the amount of time sleeping. by @fyellin in https://github.com/pygfx/wgpu-py/pull/655
* Improve performance for 'bitmap' render method by @almarklein in https://github.com/pygfx/wgpu-py/pull/680

Other:

* Stop setting AA_EnableHighDpiScaling since Qt Says it is not needed by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/657
* Remove unneeded pytest-anyio package by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/658
* Actually run the tests instead of fully skipping them by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/659
* Improve state of the art for screenshot testing by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/555
* Use default lavapipe for screenshot by @Vipitis in https://github.com/pygfx/wgpu-py/pull/660
* Fix one typo in _helpers.py by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/661
* Fix typo in README.md by @patrini32 in https://github.com/pygfx/wgpu-py/pull/662
* Improve docs regarding drives on Linux by @almarklein in https://github.com/pygfx/wgpu-py/pull/668
* Add patches for conda compatibility by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/666
* Allow disabling downloading lib when creating wheel by @almarklein in https://github.com/pygfx/wgpu-py/pull/669
* examples/imgui_cmap_picker.py - update for imgui bundle 1.6.2 by @pthom in https://github.com/pygfx/wgpu-py/pull/671


### [v0.19.3] - 10-12-2024

Fixed:

* Ensure that wgpu is compatible both with imgui 1.6.0 and older by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/649
* Clean up async code and add proper trio / rendercanvas support by @fyellin in https://github.com/pygfx/wgpu-py/pull/631
* Remove timeout in awaitable by @almarklein in https://github.com/pygfx/wgpu-py/pull/651
* Allow ``rendercanvas.get_context()`` before a backend is selected by @almarklein in https://github.com/pygfx/wgpu-py/pull/652


### [v0.19.2] - 25-11-2024

Changed:

* Update to Imgui 1.6+ by @panxinmiao in https://github.com/pygfx/wgpu-py/pull/645


### [v0.19.1] - 19-11-2024

Some internal refactoring, fix the doc theme, and compatibility with rendercanvas.

Changed:

* Do not confound STDERR and STDOUT by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/638
* More consolidation by @fyellin in https://github.com/pygfx/wgpu-py/pull/641
* Use rtd theme for docs by @almarklein in https://github.com/pygfx/wgpu-py/pull/639
* Refactor present-method mechanic by @almarklein in https://github.com/pygfx/wgpu-py/pull/642


### [v0.19.0] - 04-11-2024

Added:

* Overrideable constants by @fyellin in https://github.com/pygfx/wgpu-py/pull/579
* Implemention for multi-draw features by @fyellin in https://github.com/pygfx/wgpu-py/pull/583
* Statistics query by @fyellin in https://github.com/pygfx/wgpu-py/pull/605
* Examples for asyncio and trio by @almarklein in https://github.com/pygfx/wgpu-py/pull/608
* Add example that uses PySide6 with asyncio compat by @almarklein in https://github.com/pygfx/wgpu-py/pull/612
* Add pre-commit hooks for ruff by @claydugo in https://github.com/pygfx/wgpu-py/pull/629
* Include pre-commit in optional dependencies by @claydugo in https://github.com/pygfx/wgpu-py/pull/630

Changed:

* Async API by @almarklein in https://github.com/pygfx/wgpu-py/pull/598
  * Renamed `request_adapter()` to `request_adapter_sync()` (old method still works with warning).
  * Renamed `enumerate_adapters()` to `enumerate_adapters_sync()` (old method still works with warning).
  * Renamed `request_device()` to `request_device_sync()` (old method still works with warning).
  * Renamed `buffer.map()` to `buffer.map_sync()` (old method still works with warning).
* Renamed `request_device_trace` -> `request_device` by @fyellin in https://github.com/pygfx/wgpu-py/pull/589
* Make depth_stencil_attachment follow the spec by @fyellin in https://github.com/pygfx/wgpu-py/pull/611
* If data in `create_buffer_with_data` isn't a multiple of 4, just round up. by @fyellin in https://github.com/pygfx/wgpu-py/pull/626
* Drop support for Python 3.8.

Fixed:

* Fix snake case for 1d/2d/3d suffix by @almarklein in https://github.com/pygfx/wgpu-py/pull/617

Docs:

* Add note on examples by @almarklein in https://github.com/pygfx/wgpu-py/pull/597
* Tweaks to examples by @almarklein in https://github.com/pygfx/wgpu-py/pull/610
* Add extra version info on dev installs by @almarklein in https://github.com/pygfx/wgpu-py/pull/623
* Fix small typo in README intro by @otterbotter in https://github.com/pygfx/wgpu-py/pull/625

Internal changes:

* Remove _release code that's duplicated 19 times. by @fyellin in https://github.com/pygfx/wgpu-py/pull/590
* Fix some small errors by @fyellin in https://github.com/pygfx/wgpu-py/pull/591
* Refactor build system by @almarklein in https://github.com/pygfx/wgpu-py/pull/596
* Use ruff instead of black and flake8 by @almarklein in https://github.com/pygfx/wgpu-py/pull/599
* Fixes to testing by @almarklein in https://github.com/pygfx/wgpu-py/pull/604
* Remove stuff that's been deprecated for a while by @almarklein in https://github.com/pygfx/wgpu-py/pull/607
* Improve generated type hints and defaults by @almarklein in https://github.com/pygfx/wgpu-py/pull/606
* Add better async support for wgpu-native by @almarklein in https://github.com/pygfx/wgpu-py/pull/609
* Nitpicky changes flagged by PyCharm by @fyellin in https://github.com/pygfx/wgpu-py/pull/613
* Python 3.13.  Remove ruff flakiness by @fyellin in https://github.com/pygfx/wgpu-py/pull/614
* Structure checking by @fyellin in https://github.com/pygfx/wgpu-py/pull/615
* Eliminate warning in test suite by @fyellin in https://github.com/pygfx/wgpu-py/pull/621
* Fix install of mesa drivers by @almarklein in https://github.com/pygfx/wgpu-py/pull/622
* Only keep alive those objects that are necessary. by @fyellin in https://github.com/pygfx/wgpu-py/pull/628
* Refactor canvas context to allow presenting as image by @almarklein in https://github.com/pygfx/wgpu-py/pull/586
* Prevent extra errors when canvascontext creation fails by @almarklein in https://github.com/pygfx/wgpu-py/pull/635


### [v0.18.1] - 17-09-2024

Added:

* Add support for push constants (a wgpu native extension).

Fixed:

* Fixed jupyter gui backend.


### [v0.18.0] - 16-09-2024

Added:

* Now also building wheels for Windows aarch64.

Changed:

* Updated to latest wgpu-native (v22.1.0.5).

Fixed:

* Various typos in the docs.
* Fixed the wheels for 32bit Windows.


### [v0.17.3] - 10-09-2024

Added:

* Support for occlusion queries.
* Support for render bundles.
* Support for IMGUI, via `wgpu.utils.imgui`.
* Wx is now a fully supported GUI backend.
* A `BaseEnum` class was added to `wgpu.utils`, so it can be used in downstream libs like pygfx.
* The `WGPUCanvas.add_event_handler()` method now has an `order` arg.

Changed:

* The flags and enums are implemented using a new enum class, enabling better static code analysis (i.e. autocompletion in IDE's).
* Native (desktop) features must now be specified in the same way as normal (WebGPU) features: lowercase and with hyphens between the words.
* Bindings can omit offset and size (the full size will be used). This makes our API follow WebGPU better.
* Support omitting fields from `BindGroupLayoutEntry`, `BufferBindingLayout`, `SamplerBindingLayout`, `StorageTextureBindingLayout`, `VertexState`. See https://github.com/pygfx/wgpu-py/pull/534 for details.
* In cases where a `view_dimension` is given, it must be provided as a string (e.g. '2d'). Ints are no longer allowed, because e.g. 2 does *not* mean '2d', which can be a source of confusion.

(Due to problems with the CD process, we had to bump the version a few times.)


### [v0.16.0] - 13-06-2024

Changed:

* Updated to wgpu-native 0.19.4.1. Does not incur API changes.
* Updated to latest IDL.
* Replaced the `adapter.request_adapter_info()` method with the `adapter.info` property.


Changed:

* The `Adapter.request_adapter_info()` method is replaced by the `.info` property.


### [v0.15.3] - 12-06-2024

Added:

* Implement `canvas.set_title()` by @almarklein in https://github.com/pygfx/wgpu-py/pull/508
* Add experimental support for `get_mapped_range()` by @almarklein in https://github.com/pygfx/wgpu-py/pull/522

Changed:

* Allow `create_render_pipeline()` to use `AutoLayout` by @fyellin in https://github.com/pygfx/wgpu-py/pull/500
* Support omitting the `topology` field in the `primitive` dict passed to `create_render_pipeline()` by @fyellin in https://github.com/pygfx/wgpu-py/pull/503
* Support omitting last arguments in `set_bind_group()` by @panxinmiao in https://github.com/pygfx/wgpu-py/pull/511
* Try to optimize proxy_func by @almarklein in https://github.com/pygfx/wgpu-py/pull/507

Fixed:

* Fix wx examples by @panxinmiao in https://github.com/pygfx/wgpu-py/pull/520
* Support timestamp queryset for `begin_render_pass()` by @almarklein in https://github.com/pygfx/wgpu-py/pull/505
* Implement `destroy()` the right way by @almarklein in https://github.com/pygfx/wgpu-py/pull/519


### [v0.15.2] - 10-05-2024

Added:

* New example to demonstrate manual GLFW integration by @Korijn in https://github.com/pygfx/wgpu-py/pull/480
* More details in docs of enumerate_adapters by @almarklein in https://github.com/pygfx/wgpu-py/pull/489
* Add GPU cloud compute info to docs by @kushalkolar in https://github.com/pygfx/wgpu-py/pull/495

Changed:

* Make modifiers and buttons tuples for immutability by @hmaarrfk in https://github.com/pygfx/wgpu-py/pull/492

Fixed:

* Fix WxWgpuWindow by not calling 'add_event_handler' by @cansik in https://github.com/pygfx/wgpu-py/pull/486


### [v0.15.1] - 28-03-2024

Added:

* Users can set the `WGPU_GUI_BACKEND` environment variable to prefer a specific backend.
* Added `wgpu.gpu.enumerate_adapters()`.
* Added `adapter.summary` property, to easily get a one-line description of each adapter.

Changed:

* Avoid present-related warning messages when an exception is raised from the draw function.
* The `adapter.request_adapter_info()` now also contains `vendor_id` and `device_id`.

Fixed:

* For GUI backend selection:
  * Honor `%gui` in IPython and Jupyter.
  * Don't enter Qt event loop when Qt is active by the env (e.g. IPython).
  * Prefer glfw when we detect a running asyncio loop.
  * Make the glfw backend work in IPython.
  * The `run()` function can be called multiple times (similar to `plt.plot()`).


### [v0.15.0] - 07-03-2024

Changed:

* Updated to wgpu-native 0.19.3.1. Does not incur API changes.
* Updated to latest IDL. Does not incur API changes.

Fixed:

* Wayland is finally properly supported, albeit via XWayland.

Added:

* Tests for detecting nan and inf values in shaders.


### [v0.14.1] - 15-02-2024

Added:

* The `wgpu.DiagnosticsBase` is now a public base class that can be subclassed by downstream libs (like pygfx) to provide diagnostics.

Changed:

* Diagnostics renders bools as `✓` or `-`, and large ints get scientific notation.


### [v0.14.0] - 13-02-2024

Added:

* The feature "float32-filterable" is now available natively.
* Add list of projects using wgpu-py to [README](README.md#projects-using-wgpu-py). ([#456](https://github.com/pygfx/wgpu-py/pull/456))

Changed:

* Updated to wgpu-native 0.19.1.1. ([#458](https://github.com/pygfx/wgpu-py/pull/458))
* `Canvas.get_context()` now raises an error if no backend is selected yet (instead of returning a worthless base context object).
* Omitting the call to `context.get_current_texture()` results in a warning instead of an exception.

Removed:

* Shadertoy util is removed from the wgpu-py. It is now available as a separate package: [wgpu-shadertoy](https://github.com/pygfx/shadertoy). ([#455](https://github.com/pygfx/wgpu-py/pull/455))

Fixed:

* Devices no longer leak memory.


### [v0.13.2] - 21-12-2023

Added:

* Implement support for timestamp QuerySet.
* Add texture input and iFrameRate builtin to Shadertoy util https://github.com/pygfx/wgpu-py/pull/453


### [v0.13.1] - 08-12-2023

Fixed:

* Prevent access violation errors with GLFW on Windows.
* Prevent a segfault when deleting a `GPUPipelineLayout` (observed in a very specific use-case on LavaPipe).
* Fix `triangle_glsl.py` example.
* Fix that when logger is set to debug, errors are produced when Python exits.

Added:

* Support for linux-aarch64 (binary wheels available)! This includes Raspberry Pi's with a 64-bit OS, and adds support for building linux docker images on Apple Silicon devices without having to emulate x86 (no need for `--platform linux/amd64`).


### [v0.13.0] - 24-11-2023

Added:

* Add `iDate` builtin to Shadertoy utility.
* Allow "auto" layout args for `create_compute_pipeline()`.
* Official support for Python 3.12 and pypy.

Changed:

* Update to wgpu-native 0.18.1.2.
* `CanvasContext.get_current_texture()` now returns a `GPUTexture` instead of a `GPUTextureView`.
* `OffscreenCanvasBase.present()` now receives a `GPUTexture` instead of a `GPUTextureView`,
  and this is a new texture on each draw (no re-use).
* Renamed ``wgpu.gui.WgpuOffscreenCanvas` to `WgpuOffscreenCanvasBase`.
* The `wgpu.base` submodule that defined the GPU classes is renamed to be a private
  module. The new `wgpu.classes` namespace contains all GPU classes (and nothing else).
* The `__repr__` of the GPU classes shows a shorter canonical class name.
* Flags and Enums have a more useful `__repr__`.

Fixed:

* Dragging a window between windows with different scale factor (with Qt on Windows)
  no longer puts the window in an invalid state. A warning is still produced though.
* `GPUCommandEncoder.begin_render_pass()` binds the lifetime of passed texture views to
  the returned render pass object to prevent premature destruction when no reference to
  a texture view is kept.


### [v0.12.0] - 15-11-2023

This is a big release that contains many improvements, but also multiple API changes.

Most backward incompatible changes are due to two things: the backend
system has been refactored, making it simpler and future-proof. And we
have revised the buffer mapping API, making it more similar to the
WebGPU spec, and providing more flexible and performant ways to set
buffer data.

A summary to help you update your code:
```py
# X import wgpu.backends.rs
import wgpu


# X wgpu.request_adapter(canvas=None, power_preference="high-performance")
wgpu.gpu.request_adapter(power_preference="high-performance")

# X buffer.map_read()
buffer.map("READ")
buffer.read_mapped(...)
buffer.read_mapped(...)
buffer.unmap()

# X buffer.map_write()
buffer.map("WRITE")
buffer.write_mapped(data1, ...)
buffer.write_mapped(data2, ...)
buffer.unmap()
```

Added:

* The `wgpu.gpu` object, which represents the API entrypoint. This makes the API more clear and more similar to the WebGPU API.
* A convenience `auto` backend, and a stub `js_webgpu` backend.
* New function `enumerate_adapters()` in the `wgpu_native` backend.
* Warning about pip when wgpu-native binary is missing on Linux
* The `GPUBuffer` has new methods `map()`, `map_async()`, `unmap()`. These have been
  part of the WebGPU spec for a long time, but we had an alternative API, until now.
* The `GPUBuffer` has new methods `read_mapped()` and `write_mapped()`. These are not
  present in the WebGPU spec; they are the Pythonic alternative to `getMappedRange()`.
* Flags can now be passed as strings, and can even be combined using "MAP_READ|COPY_DIST".
* GUI events have an extra "timestamp" field, and wheel events an additional "buttons" field.
* A diagnostics subsystem that amongst other things counts GPU objects. Try e.g. `wgpu.diagnostics.print_report()`.
* Several improvements to the shadertoy util: offscreen support and a snapshot method.

Changed:

* Can create a buffer that is initially mapped: `device.create_buffer(..., mapped_at_creation=True)` is enabled again.
* The `wgpu.request_adapter()` function is moved to `wgpu.gpu.request_adapter()`. Same for the async version.
* The `canvas` argument of the `request_adapter()` function is now optional.
* The `rs` backend is renamed to `wgpu_native`.
* It is no longer necessary to explicitly import the backend.
* The `GPUDevice.request_device_tracing()` method is now a function in the `wgpu_native` backend.
* We no longer force using Vulkan on Windows. For now wgpu-native still prefers Vulkan over D3D12.
* The `wgpu.utils` subpackage is imported by default, but most submodules are not. This means that `compute_with_buffers` must be explicitly imported from `wgpu.utils.compute`.

Deprecated:

* `wgpu.request_adapter()` and its async version. Use `wgpu.gpu.request_adapter()` instead.
* The `GPUBuffer` methods `map_read()`and `map_write()` are deprecated, in favor of `map()`, `unmap()`, `read_mapped()` and `write_mapped()`.

To be clear, these are not changed:

* The convenient `device.create_buffer_with_data()` (not part of the WebGPU spec) is still available.
* The `GPUQueue.read_buffer()` and `GPUQueue.write_buffer()` methods are unchanged.

Fixed:

* The shaderutil now re-uses the default device, avoiding memoryleaks when running multiple consecutively.
* The GUI backend selection takes into account whether a backend module is already imported.
* The offscreen GUI backend no longer uses asyncio (it does not need an event loop).
* Prevent a few classes of memoryleaks. Mind that creating many `GPUDevice` objects still leaks.


### [v0.11.0] - 11-10-2023

Changed:

* Update to wgpu-native 0.17.2.1. No changes are needed in downstream code.


### [v0.10.0] - 09-10-2023

In this release the API is aligned with the latest webgpu.idl, and
we updated to wgpu-native (v0.17.0.2).

Added:

* New `wgpu.wgsl_language_features` property, which for now always returns an empty set.
* The `GPUShaderModule.compilation_info` property (and its async version) are replaced with a `get_compilation_info()` method.
* The WebGPU features "bgra8unorm-storage" and "float32-filterable" are now available.

Changed:

* The binary wheels are now based on manylinux 2.28, and the 32bit Linux wheels are no longer built.
* In WGSL: toplevel constants must be defined using `const`, using `let` will now fail.
* In WGSL: it is no longer possible to re-declare an existing variable name.
* Error messages may look a bit different, since wgpu-native now produces nice messages replacing our custom ones.
* Errors produced by a call into a wgpu-native function now produce a Python exception (no more async logging of errors).


### [v0.9.5] - 02-10-2023

Fixed:

* Fixed setting the dpi awareness in the Qt backend, by correctly looking up the Qt version.

Changed:

* Links to readthedocs now point to *stable* instead of *latest*, so that people
  reading the docs see these that reflect the latest release.
* Don't enable any features by default (previously WGPUNativeFeature_TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES was enabled).


### [v0.9.4] - 23-02-2023

Fixed:

* Fixed issue related to winid (native widgets) on embedded Qt widgets on Windows (#348).
* Fixed our example screenshot tests.


### [v0.9.3] - 20-02-2023

Changed:

* The offscreen `WgpuCanvas.draw()` method now returns a `memoryview` instead of a numpy array.
* The shadertoy util changed internally from using numpy to using a memoryview.


### [v0.9.2] - 17-02-2023

Fixed:

* Fixed that `get_preferred_format()` could crash (in `wgpuSurfaceGetSupportedFormats`) due to an upstream bug in wgpu-native (#342)

Added:

* The shadertoy util now supports GLSL, so code from the shadertoy website can be directly copied and run with wgpu (#343)


### [v0.9.1] - 13-02-2023

Changed:

* Improved documentation.

Added:

* Added `print_report()` to get a report on the internals of wgpu.
* Added `command_encoder.clear_buffer()`
* Added support for GLSL.


### [v0.9.0] - 25-01-2023

In this release the API is aligned with the latest webgpu.idl, and
we updated to the latest release of wgpu-native (v0.14.2.3).

Changed:

* To use the default `min_binding_size` in `create_bind_group_layout`, it should be `None` instead of zero.
* If the depth-stencil texture has not room for stencil data, the `stencil_read_mask` and `stencil_write_mask` fields in the `DepthStencilState` struct passed to `create_render_pipeline()` must be set to 0.
* In WGSL, `@stage(compute)` must now be `@compute`. Same for `vertex`  and `fragment`.
* In WGSL, the list of reserved words has been extended, including e.g. `mod`, `matrix` and `ref`.
* In WGSL, `smoothStep` is now `smoothstep`.

Added:

* New IDL: texture has new props `weight`, `height`, `depth_or_array_layers`.
* New IDL: Buffer has new prop `map_state`.


### [v0.8.4] - 10-01-2023

Fixed:

* The offscreen canvas's mainloop prevents leaking callbacks better (#322)
* Prevent error messages when Qt examples/apps are closed (#326)


### [v0.8.3] - 06-01-2023

Fixed:

* Prevent Qt warning about setting dpi awareness (#320)
* Make canvases close when they get deleted (#319)
* Fix qt canvas in ipython (#315)
* Make offscreen canvas closable (#309)
* Fix that the offscreen canvas had it size hardcoded, ignoring the given size (#317)
* Fixed renaming of `queue` in docs (#308)
* Fix using `.draw_frame` on qt canvas (#304)
* Add missing dev dependencies (#295)

Added:

* A shadertoy utility, plus examples (#312)

Changed:

* Improve the error prompt when wgsl code is multi line error (#311, #316)
* Tests: execute examples in the test process (#310)
* Package only the release binary (not the debug build) (#299)
* Codegen: uses in-memory file system during code generation (#303)
* Improve readme (#290, #302, #314)


### [v0.8.2] - 06-10-2022

Fixed:

* Fixed imports for PyQt6.
* Keyboard events work again for Qt 6.3.
* Fixed that overloading ``handle_event()`` did not work for a canvas based on a Qt or wx main widget/window.

Added:

* Can now add a wildcard ("*") to ``add_event_handler`` to handle all events.
* Shader error messages show more context, making shader debugging much easier.
* VSync can now be turned off to raise the frame rate when needed. Note that FPS measurements are still a poor performance benchmark!

Changed:

* GLFW canvas does not draw when minimized anymore.
* The offscreen and Jupyter canvas now use the srgb format for consistency with normal canvases.
* The examples have been adjusted for srgb colors.


### [v0.8.1] - 29-04-2022

Fixed:

* Fixed regression that `canvas.handle_event()` could no longer be overloaded to handle move and wheel events.

Changed:

* Added a note in the docs to explain that the version of the examples must match the runtime version.


### [v0.8.0] - 20-04-2022

Changed:

* Now targeting wgpu-native 0.12.0.1.
* Updated API to the latest WebGPU spec.
* Better error logging using the new callbacks in wgpu-native.
* All destructors (drop methods) are now working as they should.

To update, you need to adjust to the following API changes:

* The encoder's `end_pass()` are renamed to `end()`.
* The compute encoder's `dispatch()` is renamed `dispatch_workgroups`, and `dispatch_indirect` to `dispatch_workgroups_indirect`.
* The `load_value` is replaced with `clear_value` and `load_op`.
* Same for `depth_load_value` and `stencil_load_value`.
* The `device.create_sampler()` method for mipmap filtering now uses the `MipmapFilterMode` enum instead of the `FilterMode` enum. Since the fields of these enums are the same, you probably don't need to change anything.


To update, your shaders need the following changes:

* The decorators have changed from `[[...]]` to `@...`.
    * E.g. change `[[location(0)]]` to `@location(0)`.
    * E.g. change `[[group(0), binding(0)]]` to `@group(0) @binding(0)`.
* Structs now use `,` to separate fields instead of `;`.
* The `elseif` keyword is now `else if`.
* Buffers bound as arrays don't need to be defined via a struct anymore.


### [v0.7.7] - 12-04-2022

Fixed:

* Fixed that event handlers could not be added while in an event handler.
* Prevent swap chain errors when minimizing a window.

Added:

* The `QWgpuWidget` now also supports the autogui events.
* Our CI now tests the examples (including image comparisons).


### [v0.7.6] - 28-03-2022

Changed:

* Pointer move and wheel events are now rate-limited, leading to better performance if e.g. picking is done at each event.

Added:

* Added `wgpu.gui.base.log_exception`, a context-manager to catch and log exceptions, e.g. in event callbacks.


### [v0.7.5] - 17-03-2022

Fixed:

* Mouse down events were not emitted during double clicks in the Qt canvas.
* Mouse move events were not emitted no button is pressed in the Qt canvas.


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
  using the [jupyter_rfb event specification](https://jupyter-rfb.readthedocs.io/en/stable/events.html).
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

* The backend selection is automatic by default. To force a backend, the `WGPU_BACKEND_TYPE` environment variable can be set to e.g. "Vulkan". It could be good to do this on Windows to prevent selection of DX12 for now.


### [v0.5.3] - 04-06-2021

Added:

* `adapter.properties` now has actual values, allowing inspection of the selected
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

* Added `GPUAdapter.properties` (the amount of information it contains will increase in the future).
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
* `GPUDevice.create_bind_group_layout()` has changed the required structure of the layout entry dicts.
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

* The canvas now has a `request_draw` method.
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
