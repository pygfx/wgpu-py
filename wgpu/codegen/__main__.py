import os
import sys

from codegen.idlparser import IdlParser
from codegen.utils import blacken, to_python_name, to_neutral_name, Patcher

# todo: check code coverage and remove old code-paths
# todo: parser class attributes

root_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
resource_dir = os.path.join(root_dir, "wgpu", "resources")

# report_file = open(
#     os.path.join(resource_dir, "codegen_report.md"),
#     "wt",
#     encoding="utf-8",
#     newline="\n",
# )


ip = IdlParser(open(os.path.join(resource_dir, "webgpu.idl"), "rb").read().decode())
ip.parse(verbose=True)


##


def get_func_id_match(func_id, d):
    """Find matching func_id, taking into account sync/async method pairs."""
    for func_id_try in [func_id, func_id.replace("async", ""), func_id + "async"]:
        if func_id_try in d:
            return func_id_try


def create_py_classdef_from_idl(classname, cls):
    bases = f"({', '.join(cls.bases)})" if cls.bases else ""
    return f"class {classname}{bases}:"


def create_py_signature_from_idl(funcname, idl_line):

    preamble = "def " + to_python_name(funcname) + "("

    # Get arg names and types
    args = idl_line.split("(", 1)[1].split(")", 1)[0].split(",")
    args = [arg.strip() for arg in args if arg.strip()]
    defaults = [arg.partition("=")[2].strip() for arg in args]
    defaults = [
        default or (arg.startswith("optional ") and "None")
        for default, arg in zip(defaults, args)
    ]
    argnames = [arg.split("=")[0].split()[-1] for arg in args]
    argnames = [to_python_name(argname) for argname in argnames]
    argnames = [(f"{n}={v}" if v else n) for n, v in zip(argnames, defaults)]
    argtypes = [arg.split("=")[0].split()[-2] for arg in args]

    # Compose searches for help() call
    # todo: use?
    # searches = [func_id_match]
    # searches.extend([arg[3:] for arg in argtypes if arg.startswith("GPU")])
    # searches = [f"'{x}'" for x in sorted(set(searches))]

    # Get Python args, if one arg that is a dict, flatten dict to kwargs
    if len(argtypes) == 1 and argtypes[0].endswith(("Options", "Descriptor")):
        assert argtypes[0].startswith("GPU")
        arg_struct = ip.structs[argtypes[0][3:]]
        py_args = [field.py_arg() for field in arg_struct.values()]
        if py_args[0].startswith("label: str"):
            py_args[0] = 'label=""'
            py_args = ["self", "*"] + py_args
    else:
        py_args = ["self"] + argnames

    line = preamble + ", ".join(py_args) + "): pass\n"
    line = blacken(line, True).split("):")[0] + "):"
    return "    " + line

    # # Replace function signature
    # if "requestadapter" not in func_id:
    #     api_lines[i] = preamble + ", ".join(py_args) + "):"

    # # Insert comments
    # if fname == "base.py":
    #     api_lines.insert(i, " " * indent + "# IDL: " + idl_line)
    # api_lines.insert(
    #     i, " " * indent + f"# wgpu.help({', '.join(searches)}, dev=True)"
    # )


p = Patcher(os.path.join(root_dir, "wgpu/base.py"))

# Remove comments that we added at some point
for line, i in p.iter_lines():
    if line.lstrip().startswith(("# IDL:", "# FIXME: unknown", "# wgpu.help")):
        p.remove_line(i)


funcname_idl2py = {"constructor": "__init__"}
funcname_py2idl = {v:k for k, v in funcname_idl2py.items()}


def patch_classes(p, ip):
    classes_old = set()
    # Update existing classes in the Python code
    for classname, i1, i2 in p.iter_classes():
        classes_old.add(classname)
        if classname in ip.classes:
            old_line = p.lines[i1]
            new_line = create_py_classdef_from_idl(classname, ip.classes[classname])
            if old_line != new_line:
                fixme_line = "# FIXME: was " + old_line.split("class ", 1)[-1]
                p.replace_line(i1, f"{fixme_line}\n{new_line}")
            patch_properties(p, ip, classname, i1 + 1, i2)
            patch_methods(p, ip, classname, i1 + 1, i2)
        else:
            p.insert_line(i1, f"# FIXME: unknown class {classname}")
            print(f"Unknown class {classname}")
    # Add missing classes
    lines = []
    for classname, cls in ip.classes.items():
        if classname not in classes_old:
            new_line = create_py_classdef_from_idl(classname, cls)
            lines.append("# FIXME: new class to implement")
            lines.append(new_line)
            more_lines = []
            more_lines += get_missing_properties(ip, classname, set())
            more_lines += get_missing_methods(ip, classname, set())
            lines.extend(more_lines or ["    pass"])
    if lines:
        p.insert_line(i2 + 2, "\n".join(lines))


def patch_properties(p, ip, classname, i1, i2):
    props_old = set()
    # Update existing properties in Python code
    for propname, j1, j2 in p.iter_properties(i1):
        props_old.add(propname)
        if propname in ip.classes[classname].attributes:
            idl_line = "    # IDL: " + ip.classes[classname].attributes[propname]
            j3 = j1
            old_line = p.lines[j3]
            while "def " not in old_line:
                j3 += 1
                old_line += "\n" + p.lines[j3]
            new_line = f"    @property\n    def {propname}(self):"
            if old_line != new_line:
                fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                lines = [fixme_line, idl_line, new_line]
                p.replace_line(j1, "\n".join(lines))
                for j in range(j1 + 1, j3 + 1):
                    p.remove_line(j)
            else:
                p.insert_line(j1, idl_line)
        else:
            p.insert_line(j1, f"    # FIXME: unknown prop {classname}.{propname}")
            print(f"Unknown prop {classname}.{propname}")
    # Add missing props for this class
    lines = get_missing_properties(ip, classname, props_old)
    if lines:
        p.insert_line(i2, "\n".join(lines))


def get_missing_properties(ip, classname, props_old):
    lines = []
    for propname, idl_line in ip.classes[classname].attributes.items():
        if propname not in props_old:
            new_line = f"    @property\n    def {propname}(self):"
            lines.extend(
                [
                    "    # FIXME: new prop to implement",
                    "    # IDL: " + idl_line,
                    new_line + "\n        pass\n",
                ]
            )
    return lines


def patch_methods(p, ip, classname, i1, i2):
    funcs_old = set()
    # Update existing functions in Python code
    for funcname, j1, j2 in p.iter_methods(i1):
        funcname_idl = funcname_py2idl.get(funcname, funcname)
        funcs_old.add(funcname)
        if funcname_idl in ip.classes[classname].functions:
            idl_line = "    # IDL: " + ip.classes[classname].functions[funcname_idl]
            old_line = p.lines[j1]
            new_line = create_py_signature_from_idl(funcname, idl_line)
            if old_line != new_line:
                fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                lines = [fixme_line, idl_line, new_line]
                p.replace_line(j1, "\n".join(lines))
            else:
                p.insert_line(j1, idl_line)
        elif not funcname.startswith("_"):
            p.insert_line(j1, f"    # FIXME: unknown method {classname}.{funcname}")
            print(f"Unknown method {classname}.{funcname}")
    # Add missing functions for this class
    lines = get_missing_methods(ip, classname, funcs_old)
    if lines:
        p.insert_line(i2, "\n".join(lines))


def get_missing_methods(ip, classname, funcs_old):
    lines = []
    for funcname_idl, idl_line in ip.classes[classname].functions.items():
        funcname = funcname_idl2py.get(funcname_idl, funcname_idl)
        if funcname not in funcs_old:
            new_line = create_py_signature_from_idl(funcname, idl_line)
            lines.extend(
                [
                    "    # FIXME: new method to implement",
                    "    # IDL: " + idl_line,
                    new_line + "\n        pass\n",
                ]
            )
    return lines


patch_classes(p, ip)
# print(p.dumps())
1 / 0

##
def patch(filename, classes, structs):
    print(f"\n### Check functions in {fname}")

    starts = "# IDL: ", "# wgpu.help("
    with open(filename, "rb") as f:
        code = f.read().decode()
        api_lines = blacken(code, True).splitlines()  # inf line lenght
    api_lines = [
        line.rstrip() for line in api_lines if not line.lstrip().startswith(starts)
    ]
    api_lines.append("")

    # Detect api functions
    api_functions = {}
    current_class = None
    for i, line in enumerate(api_lines):
        if line.startswith("class "):
            current_class = line.split(":")[0].split("(")[0].split()[-1]
        if line.lstrip().startswith(("def ", "async def")):
            indent = len(line) - len(line.lstrip())
            funcname = line.split("(")[0].split()[-1]
            if not funcname.startswith("_"):
                if not api_lines[i - 1].lstrip().startswith("@property"):
                    func_id = funcname
                    funcname = to_python_name(funcname)
                    if indent:
                        func_id = current_class + "." + func_id
                    func_id = to_neutral_name(func_id)
                    api_functions[func_id] = funcname, i, indent

    # Inject IDL definitions
    count = 0
    for func_id in reversed(list(api_functions.keys())):
        func_id_match = get_func_id_match(func_id, ip.functions)
        if func_id_match:
            count += 1

            # Get info
            funcname, i, indent = api_functions[func_id]
            py_line = api_lines[i]
            idl_line = ip.functions[func_id_match]
            preamble = py_line.split("def ")[0] + "def " + funcname + "("

            # Get arg names and types
            args = idl_line.split("(", 1)[1].split(")", 1)[0].split(",")
            args = [arg.strip() for arg in args if arg.strip()]
            defaults = [arg.partition("=")[2].strip() for arg in args]
            defaults = [
                default or (arg.startswith("optional ") and "None")
                for default, arg in zip(defaults, args)
            ]
            argnames = [arg.split("=")[0].split()[-1] for arg in args]
            argnames = [to_python_name(argname) for argname in argnames]
            argnames = [(f"{n}={v}" if v else n) for n, v in zip(argnames, defaults)]
            argtypes = [arg.split("=")[0].split()[-2] for arg in args]

            # Compose searches for help() call
            searches = [func_id_match]
            searches.extend([arg[3:] for arg in argtypes if arg.startswith("GPU")])
            searches = [f"'{x}'" for x in sorted(set(searches))]

            # Get Python args, if one arg that is a dict, flatten dict to kwargs
            if len(argtypes) == 1 and argtypes[0].endswith(("Options", "Descriptor")):
                assert argtypes[0].startswith("GPU")
                arg_struct = ip.structs[argtypes[0][3:]]
                py_args = [field.py_arg() for field in arg_struct.values()]
                if py_args[0].startswith("label: str"):
                    py_args[0] = 'label=""'
                    py_args = ["self", "*"] + py_args
            else:
                py_args = ["self"] + argnames

            # Replace function signature
            if "requestadapter" not in func_id:
                api_lines[i] = preamble + ", ".join(py_args) + "):"

            # Insert comments
            if fname == "base.py":
                api_lines.insert(i, " " * indent + "# IDL: " + idl_line)
            api_lines.insert(
                i, " " * indent + f"# wgpu.help({', '.join(searches)}, dev=True)"
            )

    # Report missing
    print(f"Found {count} functions already implemented")
    for func_id in ip.functions:
        if not get_func_id_match(func_id, api_functions):
            if not (func_id.endswith("constructor") or func_id.startswith("canvas")):
                print(f"Not implemented: {ip.functions[func_id]} ({func_id})")
    for func_id in api_functions:
        if not get_func_id_match(func_id, ip.functions):
            if func_id not in ("newstruct", "getsurfaceidfromcanvas"):
                funcname = api_functions[func_id][0]
                print(f"Found unknown function {funcname} ({func_id})")


def write_base_api(classes, structs, flags, enums):

    lines = {}

    for classname, cls in classes.items():

        print()
        print(classname)
        print(cls.functions.keys())


write_base_api(ip.classes, ip.structs, ip.flags, ip.enums)
