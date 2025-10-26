"""
Codegen the JS webgpu backend, based on the parsed idl.

write to the backends/js_webgpu/_api.py file.
"""

from codegen.idlparser import Attribute, get_idl_parser
from codegen.apipatcher import IdlPatcherMixin, BaseApiPatcher
from textwrap import indent


file_preamble ="""
# Auto-generated API for the JS WebGPU backend, based on the IDL.

from ... import classes, structs, enums, flags
from ...structs import ArrayLike, Sequence # for typing hints

from pyodide.ffi import to_js
from js import Uint8Array

# TODO: move this to a new _helpers.py maybe?
from ._helpers import simple_js_accessor
"""
# maybe we should also generate a __all__ list to just import the defined classes?

# TODO: the constructor often needs more args, like device hands down self
# maybe label can be done via the property?
create_template = """
def {py_method_name}(self, **kwargs):
    descriptor = structs.{py_descriptor_name}(**kwargs)
    js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
    js_obj = self._internal.{js_method_name}(js_descriptor)

    label = kwargs.pop("label", "")
    return {return_type}(js_obj, label=label, device=self)
"""

unary_template = """
def {py_method_name}(self) -> None:
    js_obj = self._internal.{js_method_name}()
"""

# TODO: this is a bit more complex but doable.
# return needs to be optional and also resolve the promise?
# TODO: with empty body looks odd :/
positional_args_template = """
{header}
    {body}
    js_obj = self._internal.{js_method_name}({js_args})
    return js_obj # TODO maybe instead? {return_type}
"""

# might require size to be calculated if None? (offset etc)
data_conversion = """
    data = memoryview({py_data}).cast("B")
    data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
    js_data = Uint8Array.new(data_size)
    js_data.assign(data)
"""


# TODO: async template?


# most likely copy and modify the code in apipatcher.py... because we hopefully need code that looks really similar to _classes.py
idl = get_idl_parser()
helper_patcher = BaseApiPatcher() # to get access to name2py_names function


def generate_js_webgpu_api() -> str:
    """Generate the JS translation API code we can autogenerate."""


    # TODO: preamble?
    output = file_preamble + "\n\n"


    # todo import our to_js converter functions from elsewhere?
    for class_name, interface in idl.classes.items():
        # write idl line, header
        # write the to_js block
        # get label (where needed?)
        # return the constructor call to the base class maybe?


        mixins = [c for c in interface.bases if c.endswith("Mixin")]
        class_header = f"class {class_name}(classes.{class_name}, {', '.join(mixins)}):"

        class_lines = []
        # TODO: can we property some of the webgpu attributes to replace the existing private mappings

        for function_name, idl_line in interface.functions.items():
            # TODO: mixin classes seem to cause double methods? should we skip them?

            return_type = idl_line.split(" ")[0] # on some parts this doesn't exist
            if "constructor" in idl_line:
                return_type = None
                # TODO: some classes need custom constructors!
                continue # skip constructors?


            # based on apipatcher.IDlCommentINjector.get_method_comment
            args = idl_line.split("(")[1].rsplit(")")[0].split(", ")
            args = [Attribute(arg) for arg in args if arg.strip()]
            py_method_name = helper_patcher.name2py_names(class_name, function_name)

            # since we only do sync variants anyways
            py_method_name = py_method_name[0] # TODO: async always special case?
            # TODO: the create_x_pipeline_async methods become the sync variant without suffix!
            if return_type and return_type.startswith("Promise<") and return_type.endswith(">"):
                return_type = return_type.split("<")[-1].rstrip(">?")

            # case 1: single argument as a descriptor (TODO: could be optional - but that should just work)
            if len(args) == 1 and args[0].typename.endswith(
                    ("Options", "Descriptor", "Configuration")
                ):
                method_string = create_template.format(
                    py_method_name=py_method_name,
                    py_descriptor_name=args[0].typename.removeprefix("GPU"),
                    js_method_name=function_name,
                    return_type=return_type if return_type else "None",
                )
                class_lines.append(method_string)

            # case 2: no arguments (and nothing to return?)
            elif (len(args) == 0 and return_type == "undefined"):
                method_string = unary_template.format(
                    py_method_name=py_method_name,
                    js_method_name=function_name,
                )
                class_lines.append(method_string)
                # TODO: return values, could be simple or complex... so might need a constructor or not at all?

            # case 3: positional arguments, some of which might need ._internal lookup or struct->to_js conversion... but not all.
            elif (len(args) > 0):

                header = helper_patcher.get_method_def(class_name, py_method_name).partition("):")[0].lstrip()
                param_list = []
                conversion_lines = []
                js_arg_list = []
                for idx, arg in enumerate(args):
                    py_name = helper_patcher.name2py_names(class_name, arg.name)[0]
                    param_list.append(py_name)
                    # if it's a GPUObject kinda thing we most likely need to call ._internal to get the correct js object
                    if arg.typename.removesuffix("?") in idl.classes:
                        # TODO: do we need to check against none for optionals?
                        # technically the our js_accessor does this lookup too?
                        conversion_lines.append(f"js_{arg.name} = {py_name}._internal")
                        js_arg_list.append(f"js_{arg.name}")
                    # TODO: sequence of complex type?

                    elif arg.typename.removeprefix('GPU').removesuffix("?") in idl.structs:
                        conversion_lines.append(f"{py_name}_desc = structs.{arg.typename.removeprefix('GPU').removesuffix('?')}(**{py_name})")
                        conversion_lines.append(f"js_{arg.name} = to_js({py_name}_desc, eager_converter=simple_js_accessor)")
                        js_arg_list.append(f"js_{arg.name}")
                    elif py_name.endswith("data"): # maybe not an exhaustive check?
                        conversion_lines.append(data_conversion.format(py_data=py_name))
                        js_arg_list.append("js_data") #might be a problem if there is two!
                    else:
                        py_type = idl.resolve_type(arg.typename)
                        if py_type not in __builtins__ and not py_type.startswith(("enums.", "flags.")):
                            conversion_lines.append(f"# TODO: argument {py_name} of JS type {arg.typename}, py type {py_type} might need conversion")
                        js_arg_list.append(py_name)

                method_string = positional_args_template.format(
                    header=header,
                    body=("\n    ".join(conversion_lines)),
                    js_method_name=function_name,
                    js_args=", ".join(js_arg_list),
                    return_type=return_type if return_type != "undefined" else "None",
                )
                class_lines.append(method_string)

            else:
                class_lines.append(f"\n# TODO: implement codegen for {function_name} with args {args} or return type {return_type}")

        # do we need them in the first place?
        if all(line.lstrip().startswith("# TODO") for line in class_lines):
            class_lines.append("\npass")

        output += class_header
        output += indent("".join(class_lines), "    ")
        output += "\n\n" # separation between classes

    # TODO: most likely better to return a structure like
    # dict(class: dict(method : code_lines))

    return output


# TODO: we need to add some of the apidiff functions too... but I am not yet sure if we want to generate them or maybe import them?
