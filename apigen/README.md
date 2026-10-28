# wgpu APIGEN

## Introduction

The purpose of this code is to generate the wgpu-py base API, and the API of its backends.
The API is derived from the official WebGPU spec (webgpu.idl), but changed a bit to make it
fit Python better.

The API is stored in a structured data format (see below). This makes it straightforward to produce,
and also to generate the actual Python API from it.

Documentation is also present for every object in the API.

Each backend can define additional API surface, but uses suffixes for anything that is added.


## Code organization

The apigen consists of multiple "actions". Each such action is represented by a module,
which has a function with the same name that performs the action. These modules can also
be executed as a script, which is handy during development.

The names of the actions are typically "verb-noun". Call me crazy but I named them such that
when sorted alphabetically, the order of the actions is correct :)

There is also one utility module that all modules use.

There is a CLI for ease of use, and to run the whole codegen pipeline with one command.


## The steps that this code offers

* download_idl: download the latest webgpu.idl, this is only done manually from time to time.
* extract_idl: parse the IDL to produce 'wgpu_idl.json'.
* format_json: formats the static definitions.
* merge_json: combines multiple definitions files into 'wgpu_api.json'.
* patch_api: produce the classes and functions in _classes.py and all backends.
* write_api: produce API for flags, enums and structs.


## The API format structure

The API is stored in a "json-able" data structure that can be easily loaded and manipulated.
This allows generating an API definition that is precise and sort of human readable, plus
we can easily combine multiple structures, e.g. diff definitions, backend-specific additions,
and docs from multiple sources.

The structure has some two dozen leaf nodes and looks a bit like this:

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
