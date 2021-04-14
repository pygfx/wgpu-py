# WGPU-py codegen

## Introduction

The purpose of this helper package is to:

* Make maintaining wgpu-py as easy as possible;
* In particular the process of updating to new versions of WebGPU and wgpu-native;
* To validate that our API matches the WebGPU spec, and know where it differs.
* To validate that our calls into wgpu-native are correct.

We try to hit a balance between automatic code generation and proving
hints to help with manual updating. It should *not* be necessarry to check
the diffs of `webgpu.idl` or `wgpu.h`; any relevant differences should
result in changes (of code or annotations) in the respective `.py`
files. That said, during development it can be helpful to use the
WebGPU spec and the header file as a reference.

This package is *not* part of the wgpu-lib - it is a tool to help
maintain it. It has its own tests, which try to cover the utils well,
but the parsers and generators are less important to fully cover by
tests, because we are the only users. If it breaks, we fix it.


## Links

* Spec and IDL: https://gpuweb.github.io/gpuweb/
* C header: https://github.com/gfx-rs/wgpu/blob/master/ffi/wgpu.h


## Types of work that the codegen does

* Fully automatically generate modules (e.g. for constants and mappings).
* Patch manually written code:
    * Insert annotation-comments with IDL or C definitions.
    * Insert FIXME comments for code that is new or wrong.
* Generate a summarizing report. This report also contains information about
  flag/enum mismatches between IDL and wgpu.h.


## Updating the base API

The WebGPU API is specified by `webgpu.idl` (in the resources directory).
We parse this file with a custom parser (`idlparser.py`) to obtain a description
of the interfaces, enums, and flags.

The Python implementation of the flags, enums and structs is automatically generated.
Next, the Python base API (`base.py`) is updated:

* Add missing classes, methods and properties, along with a FIXME comment.
* Modify changed signatures, along with a FIXME comment.
* Put a comment that contains the corresponding IDL-line for each method and attribute.
* Mark unknown classes, methods and properties with a FIXME comment.

The update process to follow:

* Download the latest `idlparser.py`.
* Run `python codegen` to apply the automatic patches to the code.
* Now go through all FIXME comments that were added, and apply any necessary
  changes. Remove the FIXME comment if no further action is needed. Note that all
  new classes/methods/properties (instead those marked as hidden) need a docstring.
* Also check the diff of `flags.py`, `enums.py`, `structs.py` for any changes that might need manual work.
* Run `python wgpu.codegen` again to validate that all is well.
* Make a summary of the API changes to put in the release notes.
* Update downstream code, like our own tests and examples, but also e.g. pygfx.

In some cases we may want to deviate from the WebGPU API, because well ... Python
is not JavaScript. To tell the patcher logic how we deviate from the WebGPU spec:

* Decorate a method with `@apidiff.hide` to mark it as not supported by our API.
* Decorate a method with `@apidiff.add` to mark it as intended even though it does not
  match the WebGPU spec.
* Decorate a method with `@apidiff.change` to mark that our method has a different signature.


## Updating the API of the backend implementations

The backend implementations of the API (e.g. `rs.py`) are also patched.
In this case the source is the base API (instead of the IDL).

The update process is similar to the generation of the base API, except
that methods are only added if they `raise NotImplementedError()` in
the base implementation. Another difference is that this API should not
deviate from the base API - only additions are allowed (which should
be used sparingly).

You'd typically update the backends while you're updating `base.py`.


## Updating the Rust backend (`rs.py`)

The `rs.py` backend calls into a C library (wgpu-native). The codegen
helps here, by parsing the corresponding `wgpu.h` and:

* Detect and report missing flags and flag fields.
* Detect and report missing enums and enum fields.
* Generate mappings for enum field names to ints.
* Validate and annotate struct creations.
* Validate and annotate function calls into the lib.

The update process to follow:

* Download the latest `wgpu.h`.
* Run `python codegen` to generate code, apply patches, and produce a report.
* Diff the report for new differences to take into account.
* Diff `rs.py` to see what structs and functions have changed. Lines
  marked with a FIXME comment should be fixed. Others may or may not.
  Use `wgpu.h` as a reference to check available functions and structs.
* Run `python wgpu.codegen` again to validate that all is well.
* Make sure that the tests run and provide full coverage.
* This process typically does not introduce changes to the API, but certain
  features that were previously not supported could be after an update.


## Further tips

* It's probably easier to update relatively often, so that each increment is small.
* Sometimes certain features or changes are present in WebGPU, but not in wgpu-native. This may result in some manual mappings etc. which make the code less elegant. These hacks are generally temporary though.
* It's generally recommended to update `webgpu.idl` and `wgpu.h` separately. Though it could also be advantageous to combine them, to avoid the hacky stuff mentioned in the previous point.
