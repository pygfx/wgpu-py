## The API format structure 

The API is stored in a "json-able" data structure that can be easily loaded and manipulated.
This allows generating an API definition that is precise and sort of human readable, plus
we can easily combine multiple structures, e.g. diff definitions, backend-specific additions,
and docs from multiple sources. 

The structure has some two dozen leaf nodes:
```
flags.SomeFlag.doc : str
flags.SomeFlag.fields.SOME_FIELD.doc : str
flags.SomeFlag.fields.SOME_FIELD.value : int

enums.SomeEnum.doc : str
enums.SomeEnum.fields.some_field.doc : str
enums.SomeEnum.fields.some_field.value : str

structs.SomeStruct.doc : str
structs.SomeStruct.fields.some_field.doc : str
structs.SomeStruct.fields.some_field.type : str
structs.SomeStruct.fields.some_field.default : str
structs.SomeStruct.fields.some_field.required : bool

classes.GPUSomeClass.doc : str
classes.GPUSomeClass.bases : list[str]

classes.GPUSomeClass.properties.some_prop.doc : str
classes.GPUSomeClass.properties.some_prop.type : str

classes.GPUSomeClass.methods.some_method.doc : str
classes.GPUSomeClass.methods.some_method.returns.doc : str
classes.GPUSomeClass.methods.some_method.returns.type : str
classes.GPUSomeClass.methods.some_method.args.some_arg.doc : str
classes.GPUSomeClass.methods.some_method.args.some_arg.type : str
classes.GPUSomeClass.methods.some_method.args.some_arg.default : str
classes.GPUSomeClass.methods.some_method.args.some_arg.required : bool
```

As a structure, this looks a bit look this:

```
flags = {

  SomeFlag: {
    doc = ""
    fields = {
      SOME_FIELD = {
        doc = ""
        value = 1
      }
      ...
    }
  }
  ...
}


enums = {

  SomeEnum: {
    doc = ""
    fields = {
      some_field = {
        doc = ""
        value = "some-field"
      }
      ...
    }
  }
  ...
}


structs = {

  SomeStruct: {
    doc = ""
    fields = {
      some_field = {
        doc = ""
        type = "int"
        default = "0"
        required = False
      }
      ...
    }
  }
  ...
}


classes = {

  GPUSomeClass = {
    doc = ""
    bases = ["GPUObject"]
    properties = {
      some_prop = {
        doc = ""
        type = "int"
      }
      ...
    }
    methods = {
      some_method = {
        doc = ""
        returns = {
          doc = ""
          type = "None"
        }
        args = {
           some_arg {
              doc = ""
              type = ""
              default = ""
              required = True
           }
           ...
        }
      }
      ...
    }
  
  }
  ...
}

```

We could have used JSON to store these data formats, but we use TOML
because it makes it more readable and the use of sections allows for more concise definitions.
