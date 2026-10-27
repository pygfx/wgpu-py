from apigen_utils import read_definition_file, write_file

preamble = '''
"""
JS WebGPU API, but in Python. This is what the Pyodide backend should talk to.
This code is generated from webgpu.json, which is generated from webgpu.idl.
This helps validate the Pyodide backend via MyPy.
"""
'''


def write_js_api():
    """Write the Python stub API for WebGPU, to use in typechecking for web backends."""
    webgpu = read_definition_file("webgpu.json")

    lines = []

    for class_name, class_dict in webgpu["classes"].items():
        # Write classdef
        bases = class_dict.get("bases", [])
        bases_str = ""
        if bases:
            bases_str = "(" + ", ".join(bases) + ")"
        lines.append(f"class {class_name}{bases_str}:")

        # Props
        for prop_name, prop_dict in class_dict["properties"].items():
            lines.append("    @property")
            lines.append(f"    def {prop_name}(self) -> {prop_dict['type']}:")
            lines.append("        ...")
            lines.append("")

        # Methods
        for method_name, method_dict in class_dict["methods"].items():
            args = ["self"]
            for arg_name, arg_dict in method_dict["args"].items():
                arg = f"{arg_name}: {arg_dict['type']}"
                if arg_dict["default"]:
                    arg += f" = {arg_dict['default']}"
                args.append(arg)
            sig = ", ".join(args)
            lines.append(
                f"    def {method_name}({sig}) -> {method_dict['returns']['type']}:"
            )
            lines.append("        ...")
            lines.append("")

        if not (class_dict["properties"] or class_dict["methods"]):
            lines.append("    pass")
            lines.append("")

        lines.append("")

    code = preamble + "\n".join(lines)

    write_file(code, "wgpu", "backends", "js_webgpu", "webgpu_stub.py")


if __name__ == "__main__":
    write_js_api()
