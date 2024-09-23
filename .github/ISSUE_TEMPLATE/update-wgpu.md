---
name: Update wgpu
about: A checklist for updating to a newer idl and wgpu-native
title: Update wgpu
labels: ''
assignees: ''

---

For context, see the [codegen readme](https://github.com/pygfx/wgpu-py/blob/main/codegen/README.md).

## Preparations

* [ ] Perhaps warp up outstanding PR's.
* [ ] Might be a good time to roll a release of wgpu-py.
* [ ] Perhaps for pygfx too.

## Upstream

* [ ] wgpu-native is updated to a recent wgpu-core.
* [ ] wgpu-native uses a recent `webgpu.h`.
* [ ] wgpu-native has a release with these changes.

## Update to latest IDL

*Optional, can also only update wgpu-native.*

*The lines below can be copied in the PR's top post.*

* [ ] Download the latest `webgpu.idl` and place in the resources folder.
* [ ] Run `python codegen` to apply the automatic patches to the code.
* [ ] It may be necessary to tweak the `idlparser.py` to adjust to new formatting.
* [ ] Check the diff of `flags.py`, `enums.py`, `structs.py` for any changes that might need manual work.
 * [ ] Go through all FIXME comments that were added in `_classes.py`:
        * Apply any necessary changes.
        * Remove the FIXME comment if no further action is needed, or turn into a TODO for later.
        * Note that all new classes/methods/properties (instead those marked as hidden) need a docstring.
* [ ] Run `python codegen` again to validate that all is well. Repeat the step above if necessary.
* [ ] Make sure that the tests run and provide full coverage.
* [ ] Make sure that the examples all work.
* [ ] Make sure that pygfx works (again).
* [ ] Update release notes for API changes.

## Update to latest wgpu-native

*Optional, can also only update IDL*

*The lines below can be copied in the PR's top post.*

* [ ] Run `python tools/download_wgpu_native.py --version xx` to download the latest webgpu.h and DLL.
* [ ] Run `python codegen` to apply the automatic patches to the code.
* [ ] It may be necessary to tweak the `hparser.py` to adjust to new formatting.
* [ ] Diff the report for new differences to take into account.
* [ ] Diff `wgpu_native/_api.py` to get an idea of what structs and functions have changed.
* [ ] Go through all FIXME comments that were added in `_api.py`:
        * Apply any necessary changes.
        * Remove the FIXME comment if no further action is needed, or turn into a TODO for later.
* [ ] Run `python codegen` again to validate that all is well. Repeat the steps above if necessary.
* [ ] Make sure that the tests run and provide full coverage.
* [ ] Make sure that the examples all work.
* [ ] Make sure that pygfx works (again).
* [ ] Update release notes for changes in behavior.

## Wrapping up

* [ ] Update pygfx.
* [ ] Release new wgpu-py.
* [ ] Release new pygfx.
* [ ] This can be a good moment to deal with certain issues, or perhaps some issues can be closed.
