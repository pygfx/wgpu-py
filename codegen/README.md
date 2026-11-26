# wgpu-py CODEGEN

## Introduction

The wgpu-py library provides a Pythonic interpretation of the [WebGPU API](https://www.w3.org/TR/webgpu/). It is derived from the official `webgpu.idl`, but changed slightly to make it fit Python better.

That API is relatively large, and implementations use subsequently large API's. Maintaining such code by hand would be a big burden and prone to errors; we want the APIs to be precise and the code validated. On the other hand, applying fully automated code generation is also not feasible, because of the many edge-cases.  

Therefore we take a hybrid approach; some modules are completely generated, while others are 'patched'; as the IDL spec changes, classes and methods are added, marked as unknown, and signatures updated.

The purpose of the `codegen` is to:

* Generate wgpu's base API.
* Generate the API for the backend implementations.
* Validate the implementation of the backends.
* Make this process precise, yet (relatively) easy.

Note that the `codegen` package is *not* part of the wgpu library - it is a tool to help maintain it. It has its own tests.


## Updating the base API

### Introduction

The base api consists of:

* The patched `_classes.py` containing mainly abstract classes, though some properties are implemented.
* The generated `flags.py`, `enums.py` and `structs.py` define the corresponding data structures.

### How Python differs from the JavaScript API

Python is not JavaScript:

* Where in JS the input args are provided via a dict, we use kwargs directly. Nevertheless, some input args have subdicts (and sub-sub-dicts)
* For methods that are async in JavaScript (i.e return a `Promise`), we provide both an asynchronous and synchronous variant, indicated by an `_async` and `_sync` suffix.
* More specific changes are defined in `definitions/diff_base.toml`.

### Updating the IDL

To update:

* Run `python codegen --download-idl` to get the latest IDL.
* It may be necessary to update the codegen logic.
* Run `python codegen --all` to process it.
* Address the `FIXME`  comments in the source, first for the base API, then for the backends.
* Rerun the codegen as needed.

Afterwards:

* Check the diffs.
* Write API changes in the changelog.
* Make sure that the tests and examples still work.
* Make sure downstream code still works (pygfx) and make necessary updates if needed.


## Updating the wgpu-native backend

### Introducion

At the moment there is only one backend, based on [wgpu-native](https://github.com/gfx-rs/wgpu-native). We make API calls into this dynamic library, as specified by [two header files](https://github.com/gfx-rs/wgpu-native/tree/trunk/ffi). The header files can be updated independently of the IDL.

The implementation is in `wgpu/backends/wgpu_native/_api.py`. It implements additional API surface in `extras.py` which users need to explicitly import. Note that this will likely change soon, using a suffix-based extension mechanism.

The codegen also generates mappings from IDL flags to C flags (`_mappings.py`).

The majority of work in the wgpu-native backend is the conversion of  Python dicts to C structs, and then using them to call into the dynamic library. The codegen helps by validating the structs and API calls.

### Updating wgpu-native

To update:

* Download the latest `webgpu.h` and DLL using `python tools/download_wgpu_native.py --version xx`
* Run `python codegen --patch-backend-wgpu-native` to apply the automatic patches to the code.
* It may be necessary to tweak the parser to adjust to new formatting.
* Diff `wgpu_native/_api.py` to get an idea of what structs and functions have changed.
* Go through all FIXME comments that were added in `_api.py`:
* Run `python codegen --all` to validate that all is well. Repeat the steps above if necessary.

Afterwards:

* Check the diffs.
* Write API changes in the changelog.
* Make sure that the tests and examples still work.
* Make sure downstream code still works (pygfx) and make necessary updates if needed.

### Tips

* In the code, use `new_struct()` and `new_struct_p()` to create a C structure with minimal boilerplate. It also converts string enum values to their corresponding integers.
* Since the codegen adds comments for missing fields, you can instantiate a struct without any fields, then run the codegen to fill it in, and then further implement the logic.
* Use `webgpu.py` and  `wgpu.h` as a reference to check available functions and structs.
* No docstrings needed in this module.
* This process typically does not introduce changes to the API, but wgpu may now be more strict on specific usage or require changes to the shaders.
* Sometimes certain features or changes are present in WebGPU, but not in wgpu-native. This may result in some manual mappings etc. which make the code less elegant. These hacks are generally temporary though.
* It's generally recommended to update `webgpu.idl` and `webgpu.h` separately. Though it could also be advantageous to combine them, to avoid the hacky stuff mentioned in the previous point.


## Details on how the codegen works

### Introduction

The API is defines in a structured data format in JSON/TOML. This means the API
(and differences to it) can be stored and manipulated easily and explicitly.

The `webgpu.idl` is converted to `webgpu.toml`, which is diffed with `diff_base.toml` to produce `wgpu_api.toml`.
The latter has documentation for every object in the API, which is defined in `diff_docs.toml`.

Each backend can also have its own diff. It should be used sparingly and differences are marked explicitly
via a `_ext` suffix.

### Code organization

The codegen consists of multiple "actions". Each such action is represented by a module,
which has a function with the same name that performs the action. These modules can also
be executed as a script, which is handy during development.

The names of the actions are typically "verb-noun". Call me crazy but I named them such that
when sorted alphabetically, the order of the actions is correct :)

There are also some utility module that all modules use.

There is a CLI for ease of use, and to run the whole codegen pipeline with a single command.

### The steps that this code offers

* download_idl: download the latest webgpu.idl, this is only done manually from time to time.
* extract_idl: parse the 'webgpu.idl' to produce 'webgpu.toml'.
* format_defs: formats the static definitions.
* merge_defs: combines multiple definitions files into 'wgpu_api.toml'.
* patch_api: produce the classes and functions in '_classes.py'.
* patch_backend_xx: produce the API for a specific backend.
* write_api: produce API for flags, enums and structs.
* write_docs: insert / generate docs based on the api definitions, e.g. how wgpu-py differs from WebGPU.


### The API format structure

The API is stored in a "json-able" data structure that can be easily loaded and manipulated.
This allows generating an API definition that is precise and sort of human readable, plus
we can easily combine multiple structures, e.g. diff definitions, backend-specific additions,
and docs from multiple sources.

In practice we use TOML because writing *differences* to the API (additions, removals, changes)
and multiline docstrings is much friendlier in TOML.

The structure looks a bit like this:

```
{
  "flags": {
    "SomeFlag": {
      "doc": "",
      "fields": {
        "SOME_FIELD": {
          "doc": "",
          "value": 1
        }
      }
    }
  },

  "enums": {
    "SomeEnum": {
      "doc": "",
      "fields": {
        "some_field": {
          "doc": "",
          "value": "some-field"
        }
      }
    }
  },

  "structs": {
    "SomeStruct": {
      "doc": "",
      "fields": {
        "some_field": {
          "doc": "",
          "type": "int",
          "default": "0"
        }
      }
    }
  },

  "classes": {
    "GPUSomeClass": {
      "doc": "",
      "bases": ["GPUObject"],
      "properties": {
        "some_prop": {
          "doc": "",
          "type": "int"
        }
      },
      "methods": {
        "some_method": {
          "doc": "",
          "returns": {
            "doc": "",
            "type": "None"
          },
          "args": {
            "some_arg": {
              "doc": "",
              "type": "",
              "default": ""
            }
          }
        }
      }
    }
  }
}

```
