"""
Codegen the JS webgpu backend, based on the parsed idl.

write to the backends/js_webgpu/_api.py file.
"""

from codegen.idlparser import get_idl_parser


idl = get_idl_parser()

# todo import our to_js converter functions from elsewhere?

for name, interface in idl.classes.items():
    # write idl line, header
    # write the to_js block
    # get label (where needed?)
    # return the constructor call to the base class maybe?
    print(name, interface.functions)
    