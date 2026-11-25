# Code generation report

## extract_idl
* Reading codegen/definitions/webgpu.idl
* The webgpu.idl defines 43 classes with 78 methods
* The webgpu.idl defines 5 flags, 34 enums, 60 structs
* Writing codegen/definitions/webgpu.toml

## format_defs
* Reading codegen/definitions/diff_docs.toml
* Writing codegen/definitions/diff_docs.toml
* Reading codegen/definitions/diff_backend_wgpu_native.toml
* Writing codegen/definitions/diff_backend_wgpu_native.toml
* Reading codegen/definitions/diff_backend_pyodide.toml
* Writing codegen/definitions/diff_backend_pyodide.toml
* Reading codegen/definitions/diff_base.toml
* Writing codegen/definitions/diff_base.toml

## merge_defs
* Reading codegen/definitions/webgpu.toml
* Reading codegen/definitions/diff_base.toml
* Reading codegen/definitions/diff_docs.toml
* Merging
* Adding async props and methods
* Writing codegen/definitions/wgpu_api.toml
* Writing codegen/definitions/wgpu_api.json

## patch_api
* Reading codegen/definitions/wgpu_api.toml
* Found 38 classes in the base API
* Reading wgpu/_classes.py
* Writing wgpu/_classes.py

## patch_backend_wgpu_native
* Reading codegen/definitions/wgpu_api.toml
* Reading codegen/definitions/diff_backend_wgpu_native.toml
* Found 38 classes in the wgpu_native backend API
* webgpu.h/wgpu.h define 228 functions
* webgpu.h/wgpu.h define 8 flags, 70 enums, 115 structs
* Enum PipelineErrorReason missing in webgpu.h
* Enum AutoLayoutMode missing in webgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in webgpu.h
* Enum CanvasAlphaMode missing in webgpu.h
* Enum CanvasToneMappingMode missing in webgpu.h
* Mapping 267 enum mappings and 49 struct-field mappings
* Writing wgpu/backends/wgpu_native/_mappings.py
* Reading wgpu/_classes.py
* Reading wgpu/backends/wgpu_native/_api.py
* Reading wgpu/backends/wgpu_native/_api.py
* Validated 96 C structs
* Validated 153 C function calls
* Not using 86 C functions
* Writing wgpu/backends/wgpu_native/_api.py

## write_api
* Reading codegen/definitions/wgpu_api.toml
* Reading codegen/definitions/diff_backend_pyodide.toml
* Reading codegen/definitions/diff_backend_wgpu_native.toml
* Writing 5 flags
* Reading wgpu/flags.py
* Writing wgpu/flags.py
* Writing 34 enums
* Reading wgpu/enums.py
* Writing wgpu/enums.py
* Writing 57 structs
* Reading wgpu/structs.py
* Writing wgpu/structs.py

## write_docs
* Reading codegen/definitions/diff_base.toml
* Reading codegen/definitions/webgpu.toml
* Writing 44 API differences to docs/wgpu.rst
* Reading docs/wgpu.rst
* Writing docs/wgpu.rst

## write_js_api
* Reading codegen/definitions/webgpu.toml
* Writing wgpu/backends/js_webgpu/webgpu_stub.py
