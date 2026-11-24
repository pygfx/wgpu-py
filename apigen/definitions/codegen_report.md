# Code generation report

## extract_idl
* Reading webgpu.idl
* The webgpu.idl defines 43 classes with 78 methods
* The webgpu.idl defines 5 flags, 34 enums, 60 structs
* Writing webgpu.toml

## format_defs
* Reading diff_docs.toml
* Writing diff_docs.toml
* Reading diff_backend_pyodide.toml
* Writing diff_backend_pyodide.toml
* Reading diff_base.toml
* Writing diff_base.toml
* Reading diff_backend_wgpu.toml
* Writing diff_backend_wgpu.toml

## merge_defs
* Reading webgpu.toml
* Reading diff_base.toml
* Reading diff_docs.toml
* Merging
* Adding async props and methods
* Writing wgpu_api.toml
* Writing wgpu_api.json

## patch_api
* Reading wgpu_api.toml
* Reading _classes.py
* Found 40 classes
* Writing _classes.py

## write_api
* Reading wgpu_api.toml
* Reading diff_backend_pyodide.toml
* Reading diff_backend_wgpu.toml
* Writing 5 flags
* Reading flags.py
* Writing flags.py
* Writing 34 enums
* Reading enums.py
* Writing enums.py
* Writing 57 structs
* Reading structs.py
* Writing structs.py

## write_js_api
* Reading webgpu.toml
* Writing webgpu_stub.py
