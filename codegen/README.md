# wgpu-py codegen



## Introduction

### How wgpu-py is maintained

The wgpu-py library provides a Pythonic interpretation of the [WebGPU API](https://www.w3.org/TR/webgpu/). It closely follows the official spec (in the form of an [IDL file](https://gpuweb.github.io/gpuweb/webgpu.idl)). Further below is a section on how we deviate from the spec.

The actual implementation is implemented in backends. At the moment there is only one backend, based on [wgpu-native](https://github.com/gfx-rs/wgpu-native). We make API calls into this dynamic library, as specified by [two header files](https://github.com/gfx-rs/wgpu-native/tree/trunk/ffi).

The API (based on the IDL) and the backend (based on the header files) can be updated independently. In both cases, however, we are dealing with a relatively large API, which is (currently) changing quite a bit, and we need the implementation to be precise. Therefore, doing the maintenance completely by hand would be a big burden and prone to errors.

On the other hand, applying fully automated code generation is also not feasible, because of the many edge-cases that have to be taken into account. Plus the code-generation code must also be maintained.

Therefore we aim for a hybrid approach in which the aforementioned specs are used to *check* the implementations and introduce code and comments to make updates easier.

### The purpose of `codegen`

* Make maintaining wgpu-py as easy as possible;
* In particular the process of updating to new versions of WebGPU and wgpu-native;
* To validate that our API matches the WebGPU spec, and know where it differs.
* To validate that our calls into wgpu-native are correct.

During an update, it should *not* be necessary to check the diffs of `webgpu.idl` or `webgpu.h`. Instead, by running the
codegen, any relevant differences in these specs should result in changes (of code or annotations) in the respective `.py`files. That said, during development it can be helpful to use the WebGPU spec and the header files as a reference.

This package is *not* part of the wgpu library - it is a tool to help maintain it. It has its own tests, which try to cover the utils well,
but the parsers and generators are less important to fully cover by tests, because we are the only users. If it breaks, we fix it.

### General tips

* It's probably easier to update relatively often, so that each increment is small.
* Sometimes certain features or changes are present in WebGPU, but not in wgpu-native. This may result in some manual mappings etc. which make the code less elegant. These hacks are generally temporary though.
* It's generally recommended to update `webgpu.idl` and `webgpu.h` separately. Though it could also be advantageous to combine them, to avoid the hacky stuff mentioned in the previous point.



## What the codegen does in general

* Help update the front API.
  * Make changes to `_classes.py`.
  * Generate `flags.py`, `enums.py`, and `structs.py`.
*  Help update the wgpu-native backend:
  * Make changes to `backends/wgpu_native/_api.py`.
  * Generate `backends/wgpu_native/_mappings.py`.
* Write `resources/codegen_report.md`  providing a summary of the codegen process.



## Updating the front API

### Introduction

The WebGPU API is specified by `webgpu.idl` (in the resources directory). We parse this file with a custom parser (`idlparser.py`) to obtain a description of the interfaces, enums, and flags.

Note that while `wgpu/_classes.py` defines the API (and corresponding docstrings), the implementation of the majority of methods occurs in the backends, so most methods simply `raise NotimplementedError()`.

### Changes with respect to JS

In some cases we may want to deviate from the WebGPU API, because well ... Python is not JavaScript. There is a simple system in place to mark any such differences, that also makes sure that these changes are listed in the docs. To mark how the py API deviates from the WebGPU spec:

* Decorate a method with `@apidiff.hide` to mark it as not supported by our API.
* Decorate a method with `@apidiff.add` to mark it as intended even though it does not
  match the WebGPU spec.
* Decorate a method with `@apidiff.change` to mark that our method has a different signature.

Other changes include:

* Where in JS the input args are provided via a dict, we use kwargs directly. Nevertheless, some input args have subdicts (and sub-sub-dicts)
*  For methods that are async in IDL, we also provide sync methods. The Async method names have an "_async" suffix.

### Codegen summary

* Generate `flags.py`, `enums.py`, and `structs.py`.

* Make changes to `_classes.py`.

  * Add missing classes, methods and properties, along with a FIXME comment..

  * Modify changed signatures, along with a FIXME comment.
  * Mark unknown classes, methods and properties with a FIXME comment.

  * Put a comment that contains the corresponding IDL-line for each method and attribute.


### The update process

* Download the latest [webgpu.idl](https://gpuweb.github.io/gpuweb/webgpu.idl) and place in the resources folder.
* Run `python codegen` to apply the automatic patches to the code.
* It may be necessary to tweak the `idlparser.py` to adjust to new formatting.
* Check the diff of `flags.py`, `enums.py`, `structs.py` for any changes that might need manual work.
* Go through all FIXME comments that were added in `_classes.py`:
    * Apply any necessary changes.
    * Remove the FIXME comment if no further action is needed, or turn into a TODO for later.
    * Note that all new classes/methods/properties (instead those marked as hidden) need a docstring.
* Run `python codegen` again to validate that all is well. Repeat the step above if necessary.
* Make sure that the tests run and provide full coverage.
* Make sure that the examples all work.
* Update downstream code, like our own tests and examples, but also e.g. pygfx.
* Make a summary of the API changes to put in the release notes.



## Updating the wgpu-native backend

### Introduction

The backends are almost a copy of `_classes.py`: all methods in `_classes.py` that `raise NotImplementedError()` must be implemented.

The wgpu-native backend calls into a dynamic library, which interface is specified by  `webgpu.h` and `wgpu.h` (in the resources directory). We parse these files with a custom parser (`hparser.py`) to obtain a description of the interfaces, enums, flags, and structs.

The majority of work in the wgpu-native backend is the conversion of  Python dicts to C structs, and then using them to call into the dynamic library. The codegen helps by validating the structs and API calls.

### Tips

* In the code, use `new_struct()` and `new_struct_p()` to create a C structure with minimal boilerplate. It also converts string enum values to their corresponding integers.

* Since the codegen adds comments for missing fields, you can instantiate a struct without any fields, then run the codegen to fill it in, and then further implement the logic.
* The API of the backends should not deviate from the base API - only`@apidiff.add` is allowed (and should be used sparingly).
* Use `webgpu.py` and  `wgpu.h` as a reference to check available functions and structs.
* No docstrings needed in this module.
* This process typically does not introduce changes to the API, but wgpu may now be more strict on specific usage or require changes to the shaders.

### Codegen summary

* Generate `backends/wgpu_native/_mappings.py`.
  * Generate mappings for enum field names to ints.
  * Detect and report missing flags and enum fields.

* Make changes to `wgpu_native/_api.py`.
  * Validate and annotate function calls into the lib.
  * Validate and annotate struct creations (missing struct fields are filled in).
  * Ensure that each incoming struct is checked to catch invalid input.

### The update process

* Download the latest `webgpu.h` and DLL using `python download-wgpu-native.py --version xx`
* Run `python codegen` to apply the automatic patches to the code.
* It may be necessary to tweak the `hparser.py` to adjust to new formatting.
* Diff the report for new differences to take into account.
* Diff `wgpu_native/_api.py` to get an idea of what structs and functions have changed.
* Go through all FIXME comments that were added in `_api.py`:
  *  Apply any necessary changes.
  * Remove the FIXME comment if no further action is needed, or turn into a TODO for later.

* Run `python codegen` again to validate that all is well. Repeat the steps above if necessary.
* Make sure that the tests run and provide full coverage.
* Make sure that the examples all work.
* Update downstream code, like our own tests and examples, but also e.g. pygfx.

* Make a summary of the API changes to put in the release notes.
