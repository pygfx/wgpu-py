"""
Provide functionality to generate/patch a Python API based on the WebGPU spec (IDL).
"""

from .idlparser import IdlParser
from .utils import blacken, to_python_name, to_neutral_name, Patcher


_map_funcname_idl2py = {"constructor": "__init__"}
_map_funcname_py2idl = {v: k for k, v in _map_funcname_idl2py.items()}


def funcname_py2idl(name):
    return _map_funcname_py2idl.get(name, name)


def funcname_idl2py(name):
    return _map_funcname_idl2py.get(name, name)


def propname_py2idl(name):
    is_capital = False
    name2 = ""
    for c in name:
        if c == "_":
            is_capital = True
        elif is_capital:
            name2 += c.upper()
            is_capital = False
        else:
            name2 += c
    return name2


def propname_idl2py(name):
    name2 = ""
    for c in name:
        c_ = c.lower()
        if c_ != c:
            name2 += "_"
        name2 += c_
    return name2


# %%%%% Helper functions


def create_py_classdef_from_idl(classname, cls):
    # Make sure that GPUObjectBase comes last, for MRO
    bases = sorted(cls.bases or [], key=lambda n: n.count("GPUObjectBase"))
    bases = f"({', '.join(bases)})" if bases else ""
    if not bases and classname.lower().endswith("error"):
        bases = "(Exception)"
        if "memory" in classname:
            bases = "(MemoryError)"
    return f"class {classname}{bases}:"


def create_py_signature_from_idl(idl, funcname, idl_line):

    preamble = "def " + to_python_name(funcname) + "("
    if "async" in funcname:
        preamble = "async " + preamble

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
        arg_struct = idl.structs[argtypes[0][3:]]
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


# %% The patching code


def patch_module(idl, code):
    """Given the IDL and Python code, will patch the code to apply
    the API matching the IDL.
    """
    for patcher in [CommentRemover(), PatcherBasedOnIdl(idl), IdlCommentInjector(idl)]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


def patch_backend(base_code, code):
    for patcher in [CommentRemover(), PatcherBasedOnBaseAPI(base_code)]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


class CommentRemover(Patcher):
    def apply(self, code):
        self._init(code)
        for line, i in self.iter_lines():
            if line.lstrip().startswith(("# IDL:", "# FIXME: unknown", "# wgpu.help")):
                self.remove_line(i)


class BasePatcher(Patcher):
    def apply(self, code):
        self._init(code)
        self.patch_classes()

    def patch_classes(self):
        seen_classes = set()

        # Update existing classes in the Python code
        for classname, i1, i2 in self.iter_classes():
            seen_classes.add(classname)
            if self.class_is_known(classname):
                old_line = self.lines[i1]
                new_line = self.get_class_def(classname)
                if old_line != new_line:
                    fixme_line = "# FIXME: was " + old_line.split("class ", 1)[-1]
                    # self.replace_line(i1, f"{fixme_line}\n{new_line}")
                    self.replace_line(i1, f"{new_line}")
                self.patch_properties(classname, i1 + 1, i2)
                self.patch_methods(classname, i1 + 1, i2)
            else:
                self.insert_line(i1, f"# FIXME: unknown class {classname}")
                print(f"Unknown class {classname}")

        # Add missing classes
        lines = []
        for classname in self.get_class_names():
            if classname not in seen_classes:
                lines.append("# FIXME: new class to implement")
                lines.append(self.get_class_def(classname))
                more_lines = []
                more_lines += self.get_missing_properties(classname, set())
                more_lines += self.get_missing_methods(classname, set())
                lines.extend(more_lines or ["    pass"])
        if lines:
            self.insert_line(i2, "\n".join(lines))

    def patch_properties(self, classname, i1, i2):
        seen_props = set()

        # Update existing properties in Python code
        for propname, j1, j2 in self.iter_properties(i1):
            seen_props.add(propname)
            if self.prop_is_known(classname, propname):
                old_line = self.lines[j1]
                j3 = j1
                while "def " not in old_line:
                    j3 += 1
                    old_line += "\n" + self.lines[j3]
                new_line = f"    @property\n    def {propname}(self):"
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
                    for j in range(j1 + 1, j3 + 1):
                        self.remove_line(j)
            else:
                self.insert_line(
                    j1, f"    # FIXME: unknown prop {classname}.{propname}"
                )
                print(f"Unknown prop {classname}.{propname}")

        # Add missing props for this class
        lines = self.get_missing_properties(classname, seen_props)
        if lines:
            self.insert_line(i2, "\n".join(lines))

    def patch_methods(self, classname, i1, i2):
        seen_funcs = set()

        # Update existing functions in Python code
        for funcname, j1, j2 in self.iter_methods(i1):
            seen_funcs.add(funcname)
            pre_lines = "\n".join(self.lines[j1 - 3 : j1])
            if "@apidiff.add" in pre_lines:
                pass
            elif self.method_is_known(classname, funcname):
                if "@apidiff.hide" in pre_lines:
                    pass  # continue as normal
                elif "@apidiff.change" in pre_lines:
                    continue
                old_line = self.lines[j1]
                new_line = self.get_method_def(classname, funcname)
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
            elif not funcname.startswith("_"):
                self.insert_line(
                    j1, f"    # FIXME: unknown method {classname}.{funcname}"
                )
                print(f"Unknown method {classname}.{funcname}")

        # Add missing functions for this class
        lines = self.get_missing_methods(classname, seen_funcs)
        if lines:
            self.insert_line(i2, "\n".join(lines))

    def get_missing_properties(self, classname, seen_props):
        lines = []
        for propname in self.get_required_prop_names(classname):
            if propname not in seen_props:
                lines.append("    # FIXME: new prop to implement")
                lines.append("    @property")
                lines.append("    def {propname}(self):")
                lines.append("        raise NotImplementedError()")
                lines.append("")
        return lines

    def get_missing_methods(self, classname, seen_funcs):
        lines = []
        for funcname in self.get_required_method_names(classname):
            if funcname not in seen_funcs:
                lines.append("    # FIXME: new method to implement")
                lines.append(self.get_method_def(classname, funcname))
                lines.append("        raise NotImplementedError()\n")
        return lines


class PatcherBasedOnIdl(BasePatcher):
    def __init__(self, idl):
        super().__init__()
        self.idl = idl

    def class_is_known(self, classname):
        return classname in self.idl.classes

    def get_class_def(self, classname):
        return create_py_classdef_from_idl(classname, self.idl.classes[classname])

    def get_method_def(self, classname, funcname):
        functions = self.idl.classes[classname].functions
        funcname_idl = funcname_py2idl(funcname)
        if "_async" in funcname and funcname_idl not in functions:
            funcname_idl = funcname_py2idl(funcname.replace("_async", ""))
        idl_line = functions[funcname_idl]
        return create_py_signature_from_idl(self.idl, funcname, idl_line)

    def prop_is_known(self, classname, propname):
        propname_idl = propname_py2idl(propname)
        return propname_idl in self.idl.classes[classname].attributes

    def method_is_known(self, classname, funcname):
        functions = self.idl.classes[classname].functions
        funcname_idl = funcname_py2idl(funcname)
        if "_async" in funcname and funcname_idl not in functions:
            funcname_idl = funcname_py2idl(funcname.replace("_async", ""))
        return funcname_idl if funcname_idl in functions else None

    def get_class_names(self):
        return list(self.idl.classes.keys())

    def get_required_prop_names(self, classname):
        propnames_idl = self.idl.classes[classname].attributes.keys()
        return [propname_idl2py(x) for x in propnames_idl]

    def get_required_method_names(self, classname):
        funcnames_idl = self.idl.classes[classname].functions.keys()
        return [funcname_idl2py(x) for x in funcnames_idl]


class BaseCommentInjector(Patcher):
    def patch_classes(self):
        for classname, i1, i2 in self.iter_classes():
            if self.class_is_known(classname):
                comment = self.get_class_comment(classname)
                if comment:
                    self.insert_line(i1, comment)
                self.patch_properties(classname, i1 + 1, i2)
                self.patch_methods(classname, i1 + 1, i2)

    def patch_properties(self, classname, i1, i2):
        for propname, j1, j2 in self.iter_properties(i1):
            comment = self.get_prop_comment(classname, propname)
            if comment:
                self.insert_line(j1, comment)

    def patch_methods(self, classname, i1, i2):
        for funcname, j1, j2 in self.iter_methods(i1):
            comment = self.get_method_comment(classname, funcname)
            if comment:
                if self.lines[j1 - 1].lstrip().startswith("@apidiff"):
                    self.insert_line(j1 - 1, comment)
                else:
                    self.insert_line(j1, comment)


class IdlCommentInjector(BaseCommentInjector, PatcherBasedOnIdl):
    def get_class_comment(self, classname):
        return None

    def get_prop_comment(self, classname, propname):
        if self.prop_is_known(classname, propname):
            propname_idl = propname_py2idl(propname)
            return "    # IDL: " + self.idl.classes[classname].attributes[propname_idl]

    def get_method_comment(self, classname, funcname):
        funcname_idl = self.method_is_known(classname, funcname)
        if funcname_idl:
            return "    # IDL: " + self.idl.classes[classname].functions[funcname_idl]


class PatcherBasedOnBaseAPI(BasePatcher):
    def __init__(self, base_api_code):
        super().__init__()

        p1 = Patcher(base_api_code)

        # Collect what's needed
        self.classes = classes = {}
        for classname, i1, i2 in p1.iter_classes():
            methods = {}
            for funcname, j1, j2 in p1.iter_methods(i1 + 1):
                pre_lines = "\n".join(p1.lines[j1 - 3 : j1])
                if "@apidiff.hide" in pre_lines:
                    continue  # method (currently) not part of our API
                body = "\n".join(p1.lines[j1 + 1 : j2 + 1])
                must_overload = "raise NotImplementedError()" in body
                methods[funcname] = p1.lines[j1], must_overload
            classes[classname] = p1.lines[i1], methods
            # We assume that all properties can be implemented on the base class

    def class_is_known(self, classname):
        return classname in self.classes

    def get_class_def(self, classname):
        line, _ = self.classes[classname]
        if "):" in line:
            return line.replace("(", f"(base.{classname}, ")
        else:
            return line.replace(":", f"(base.{classname}):")

    def get_method_def(self, classname, funcname):
        _, methods = self.classes[classname]
        line, _ = methods[funcname]
        return line

    def prop_is_known(self, classname, propname):
        return False

    def method_is_known(self, classname, funcname):
        _, methods = self.classes[classname]
        return funcname in methods

    def get_class_names(self):
        return list(self.classes.keys())

    def get_required_prop_names(self, classname):
        return []

    def get_required_method_names(self, classname):
        _, methods = self.classes[classname]
        return list(name for name in methods.keys() if methods[name][1])
