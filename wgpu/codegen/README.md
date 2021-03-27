# WGPU-py codegen

The purpose of this submodule is to generate/patch our code, so that
updating to new versions of the WebGPU spec and wgpu-native header is
as easy as possible. We try to hit a balance between automating and
inserting hints to help with manual updating.


## API generation

The WebGPU API is specified by `webgpu.idl` (in the resources directory).
We parse this file with a custom parser (`idlparser.py`) to obtain a description
of the interfaces, enums, and flags.

Next, this information is used to generate/update the Python base API, by patching `base.py`:

* Add missing classes, methods and properties, along with a FIXME comment.
* Modify changed signatures, along with a FIXME comment.
* Put a comment that contains the corresponding IDL-line for each method and attribute.
* Mark unknown classes, methods and properties with a FIXME comment.

The update process is as follows:

* Download the latest `idlparser.py`.
* Run `python wgpu.codegen` to apply the automatic patches to the code.
* Now go through all FIXME comments that were made, and apply any necessary
  changes. Remove the FIXME comment if no further action is needed.
* Run `python wgpu.codegen` again (with a check arg?) to validate that all is well.

In some cases we may want to deviate from the WebGPU API, because well ... Python
is not JavaScript. To tell the patcher logic how we deviate from the WebGPU spec:

* Decorate a method with `@apidiff.hide` to mark it as not supported by our API.
* Decorate a method with `@apidiff.add` to mark it as proper API even though it does not
  match the WebGPU spec.
* Decorate a method with `@apidiff.change` to mark that our method has a different signature.


## Keeping backend API's up-to-date

The implementations of the API (i.e. the backends, e.g. `rs.py`) are also patched.
In this case the source is the base API. The process itself is very similar to
the generation of the base API, except that methods are only added if they
`raise NotImplementedError()` in the base implementation.

Another difference is that this API should not deviate from the base API - only
additions are (sparingly) allowed.


## Support the implementation of the Rust backend

TODO
