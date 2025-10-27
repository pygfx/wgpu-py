"""
Codegen the JS webgpu backend, based on the parsed idl.

write to the backends/js_webgpu/_api.py file.
"""

import re
from codegen.idlparser import Attribute, get_idl_parser, Interface
from codegen.apipatcher import IdlPatcherMixin, BaseApiPatcher
from codegen.utils import Patcher
from textwrap import indent, dedent


file_preamble ="""
# Auto-generated API for the JS WebGPU backend, based on the IDL and custom implementations.

from ... import classes, structs, enums, flags
from ...structs import ArrayLike, Sequence # for typing hints
from typing import Union

from pyodide.ffi import to_js, run_sync, JsProxy
from js import window, Uint8Array

from ._helpers import simple_js_accessor
from ._implementation import GPUPromise
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
    return {return_type}(label, js_obj, device=self)
"""

unary_template = """
def {py_method_name}(self) -> None:
    self._internal.{js_method_name}()
"""

# TODO: this is a bit more complex but doable.
# return needs to be optional and also resolve the promise?
# TODO: with empty body looks odd :/
positional_args_template = """
{header}
    {body}
    self._internal.{js_method_name}({js_args})
"""
# TODO: construct a return value if needed?


# might require size to be calculated if None? (offset etc)
data_conversion = """
    if {py_data} is not None:
        data = memoryview({py_data}).cast("B")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        js_data = Uint8Array.new(data_size)
        js_data.assign(data)
    else:
        js_data = None
"""

# most likely copy and modify the code in apipatcher.py... because we hopefully need code that looks really similar to _classes.py
idl = get_idl_parser()
helper_patcher = BaseApiPatcher() # to get access to name2py_names function

# can't use importlib because pyodide isn't available -.-
# maybe use ast?
custom_implementations = open("./wgpu/backends/js_webgpu/_implementation.py").read()

class JsPatcher(Patcher):
    # TODO: we can put custom methods here!
    pass

patcher = JsPatcher(custom_implementations)

def generate_method_code(class_name: str, function_name: str, idl_line: str) -> str:
    # TODO: refactor into something like this
    pass

def get_class_def(class_name: str, interface: Interface) -> str:
    # TODO: refactor
    pass


# basically three cases for methods (from idl+apidiff):
# 1. alreayd exists in _classes.py and can be used as is (generate nothing)
# 2. custom implementation in _implementations.py (copy string over)
# 3. auto-generate remaining methods based on idl



def generate_js_webgpu_api() -> str:
    """Generate the JS translation API code we can autogenerate."""


    # TODO: preamble?
    output = file_preamble + "\n\n"

    # classname, start_line, end_line
    custom_classes = {c: (s, e) for c, s, e in patcher.iter_classes()}

    # todo import our to_js converter functions from elsewhere?
    # we need to have the mixins first!
    ordered_classes = sorted(idl.classes.items(), key=lambda c: "Mixin" not in c[0]) # mixins first
    for class_name, interface in ordered_classes:
        # write idl line, header
        # write the to_js block
        # get label (where needed?)
        # return the constructor call to the base class maybe?

        custom_methods = {}

        if class_name in custom_classes:
            class_line = custom_classes[class_name][0] +1
            for method_name, start_line, end_line in patcher.iter_methods(class_line):
                # grab the actual contents ?
                # maybe include a comment that is in the line prior from _implementation.py?
                method_lines = patcher.lines[start_line:end_line+1]
                custom_methods[method_name] = method_lines

            # include custom properties too
            for prop_name, start_line, end_line in patcher.iter_properties(class_line):
                prop_lines = patcher.lines[start_line-1:end_line+1]
                custom_methods[prop_name] = prop_lines

        mixins = [c for c in interface.bases if c.endswith("Mixin")]
        class_header = f"class {class_name}(classes.{class_name}, {', '.join(mixins)}):"

        class_lines = ["\n"]
        # TODO: can we property some of the webgpu attributes to replace the existing private mappings

        for function_name, idl_line in interface.functions.items():
            return_type = idl_line.split(" ")[0] # on some parts this doesn't exist
            py_method_name = helper_patcher.name2py_names(class_name, function_name)
            # TODO: resolve async double methods!
            py_method_name = py_method_name[0] # TODO: async always special case?

            if py_method_name in custom_methods:
                # Case 2: custom implementation exists!
                class_lines.append(f"\n# Custom implementation for {function_name} from _implementation.py:\n")
                class_lines.append(dedent("\n".join(custom_methods[py_method_name])))
                class_lines.append("\n") # for space I guess
                custom_methods.pop(py_method_name) # remove ones we have added.
                continue

            if py_method_name == "__init__":
                # whacky way, but essentially this mean classes.py implements a useable constructor already.
                continue

            # TODO: mixin classes seem to cause double methods? should we skip them?

            # based on apipatcher.IDlCommentINjector.get_method_comment
            args = idl_line.split("(")[1].rsplit(")")[0].split(", ")
            args = [Attribute(arg) for arg in args if arg.strip()]

            # TODO: the create_x_pipeline_async methods become the sync variant without suffix!
            if return_type and return_type.startswith("Promise<") and return_type.endswith(">"):
                return_type = return_type.split("<")[-1].rstrip(">?")

            # skip these for now as they are more troublesome -.-
            if py_method_name.endswith("_sync"):
                class_lines.append(f"\n# TODO: {function_name} sync variant likely taken from _classes.py directly!")
                continue

            if function_name.endswith("Async"):
                class_lines.append(f"\n# TODO: was was there a redefinition for {function_name} async variant?")
                continue

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
                # put all potentially forward refrenced classes into quotes
                header = " ".join(f'"{h}"' if h.startswith("GPU") else h for h in header.split(" ")).replace(':"','":')
                # turn all optional type hints into Union with None
                # int | None -> Union[int, None]
                exp = r":\s([\w\"]+)\s\| None"
                header = re.sub(exp, lambda m: f": Union[{m.group(1)}, None]", header)
                header = header.replace('Sequence[GPURenderBundle]', 'Sequence["GPURenderBundle"]') # TODO: just a temporary bodge!

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

                    elif arg.typename.removeprefix('GPU').removesuffix("?") in idl.structs and not arg.typename == "GPUExtent3D":
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

                # TODO: have a return line constructor function?

            else:
                class_lines.append(f"\n# TODO: implement codegen for {function_name} with args {args} or return type {return_type}")

        # if there are some methods not part of the idl, we should write them too
        if custom_methods:
            class_lines.append("\n# Additional custom methods from _implementation.py:\n")
            for method_name, method_lines in custom_methods.items():
                class_lines.append(dedent("\n".join(method_lines)))
                class_lines.append("\n\n")

        # do we need them in the first place?
        if all(line.lstrip().startswith("#") for line in class_lines if line.strip()):
            class_lines.append("\npass")

        output += class_header
        output += indent("".join(class_lines), "    ")
        output += "\n\n" # separation between classes

    # TODO: most likely better to return a structure like
    # dict(class: dict(method : code_lines))


    # TODO: postamble:
    output += "\ngpu = GPU()\n"

    return output


# TODO: we need to add some of the apidiff functions too... but I am not yet sure if we want to generate them or maybe import them?
