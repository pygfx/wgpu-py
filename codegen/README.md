# WGPU-py codegen

## Introduction

The purpose of this helper package is to:

* Make maintaining wgpu-py as easy as possible;
* In particular the process of updating to new versions of WebGPU and wgpu-native;
* To validate that our API matches the WebGPU spec, and know where it differs.
* To validate that our calls into wgpu-native are correct.

We try to hit a balance between automatic code generation and proving
hints to help with manual updating.

This package is *not* part of the wgpu-lib - it is a tool to help
maintain it. It has its own tests, which try to cover the utils well,
but the parsers and generators are less important to fully cover by
tests, because we are the only users. If it breaks, we fix it.


## Updating the base API

The WebGPU API is specified by `webgpu.idl` (in the resources directory).
We parse this file with a custom parser (`idlparser.py`) to obtain a description
of the interfaces, enums, and flags.

Next, this information is used to update the Python base API in `base.py`:

* Add missing classes, methods and properties, along with a FIXME comment.
* Modify changed signatures, along with a FIXME comment.
* Put a comment that contains the corresponding IDL-line for each method and attribute.
* Mark unknown classes, methods and properties with a FIXME comment.

The update process is as follows:

* Download the latest `idlparser.py`.
* Run `python codegen` to apply the automatic patches to the code.
* Now go through all FIXME comments that were made, and apply any necessary
  changes. Remove the FIXME comment if no further action is needed. Note that all
  new classes/methods/properties (instead those marked as hidden) need a docstring.
* Run `python wgpu.codegen` again (with a check arg?) to validate that all is well.

In some cases we may want to deviate from the WebGPU API, because well ... Python
is not JavaScript. To tell the patcher logic how we deviate from the WebGPU spec:

* Decorate a method with `@apidiff.hide` to mark it as not supported by our API.
* Decorate a method with `@apidiff.add` to mark it as intended even though it does not
  match the WebGPU spec.
* Decorate a method with `@apidiff.change` to mark that our method has a different signature.


## Updating the backend implementations

The implementations of the API (i.e. the backends, e.g. `rs.py`) are also patched.
In this case the source is the base API (not IDL).

The update process is similar to the generation of the base API, except
that methods are only added if they `raise NotImplementedError()` in
the base implementation.

Another difference is that this API should not deviate from the base API - only
additions are (sparingly) allowed.


## Support the implementation of the Rust backend

TODO
