"""
Codegen the JS webgpu backend, based on the parsed idl.

write to the backends/js_webgpu/_api.py file.
"""

from codegen.idlparser import get_idl_parser
from textwrap import indent

method_template = """
def {py_method_name}(self, *args, **kwargs):
    js_args = to_js(args)
    js_kwargs = to_js(kwargs)

    js_obj = self._{js_method_name}( *js_args, **js_kwargs )

    label = js_kwargs.pop("label", "")
    return {class}(js_obj, label=label) # more kwargs? only for get_ and create_?
"""


idl = get_idl_parser()

# todo import our to_js converter functions from elsewhere?

for name, interface in idl.classes.items():
    # write idl line, header
    # write the to_js block
    # get label (where needed?)
    # return the constructor call to the base class maybe?
    print(name)
    for function in interface.functions:
        print("  ", function, type(function))
    # print(name, interface.functions)
    