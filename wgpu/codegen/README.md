# WGPU-py codegen

The purpose of this submodule is to generate/patch our code, so that
updating to new versions of the WebGPU spec and wgou-native header is
as easy as possible. We try to hit a balance between automating and
inserting hints to help with manual updating.


## API generation

The WebGPU API is specified by `webgpu.idl` (in the resources directory).
We parse this file with a custom parser (`idlparser.py`) to obtain a description
of the interfaces, enums, and flags.

Next, this information is used to generate/update the Python API, by patching a module:

* Add missing classes, methods and properties, along with a FIXME comment.
* Modify changed signatures, along with a FIXME comment.
* Put a comment that contains the corresponding IDL-line for each method and attribute.
* Mark unknown classes, methods and properties with a FIXME comment.

The update process is as follows:

* Download the latest `idlparser.py`.
* Run `python wgpu.codegen` to apply the automatic patches to the code.
* Now go through all FIXME comments that were made, and apply any necessary
  changes. When done, remove the FIXME comment.
  * In the abstract API, where we write the docstrings.
  * In the rs backend.
  * Later we may have more backends.
* Run `python wgpu.codegen` again (with a check arg?) to validate that all is well.

There may be cases where the codegen includes classes or methods that we
chose not to implement (not yet, or not ever). How to deal with these?

There may be cases where we provide additional or different methods and properties.
How to deal with these?



## C-headers

TODO