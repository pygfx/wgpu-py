"""
Provide functionality to generate/patch a Python API based on the WebGPU spec (IDL).
"""

import os

from .idlparser import IdlParser
from .utils import lib_dir, blacken, to_python_name, Patcher


def patch_base_api(code):
    """Given the Python code, applies patches to make the code conform
    to the IDL.
    """

    # Obtain the idl definition
    with open(os.path.join(lib_dir, "resources", "webgpu.idl"), "rb") as f:
        idl = IdlParser(f.read().decode())
    idl.parse(verbose=True)

    # Patch!
    for patcher in [CommentRemover(), BaseApiPatcher(idl), IdlCommentInjector(idl)]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


def patch_backend_api(code):
    """Given the Python code, applies patches to make the code conform
    to the base API.
    """

    # Obtain the base API definition
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        base_api_code = f.read().decode()

    # Patch!
    for patcher in [CommentRemover(), BackendApiPatcher(base_api_code)]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


class CommentRemover(Patcher):
    """A patcher that removes comments that we add in other parsers,
    to prevent accumulating comments.
    """

    def apply(self, code):
        self._init(code)
        for line, i in self.iter_lines():
            if line.lstrip().startswith(("# IDL:", "# FIXME: unknown", "# wgpu.help")):
                self.remove_line(i)


class AbstractCommentInjector(Patcher):
    """A base patcher that can insert  helpful comments in front of
    properties, methods, and classes. It does not mark any as new or unknown,
    since that is the task of the API patchers.
    """

    # Note that in terms of structure, this class is basically a simplified
    # version of the AbstractApiPatcher

    def apply(self, code):
        self._init(code)
        self.patch_classes()

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
        for methodname, j1, j2 in self.iter_methods(i1):
            comment = self.get_method_comment(classname, methodname)
            if comment:
                if self.lines[j1 - 1].lstrip().startswith("@apidiff"):
                    self.insert_line(j1 - 1, comment)
                else:
                    self.insert_line(j1, comment)


class AbstractApiPatcher(Patcher):
    """The base patcher to update a wgpu API.

    This code is generalized, so it can be used both to generate the base API
    as well as the backends (implementations).

    The idea is to walk over all classes, patch it if necessary, then
    walk over each of its properties and methods to patch these too.
    """

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
                    self.replace_line(i1, f"{fixme_line}\n{new_line}")
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

        # Add missing properties for this class
        lines = self.get_missing_properties(classname, seen_props)
        if lines:
            self.insert_line(i2, "\n".join(lines))

    def patch_methods(self, classname, i1, i2):
        seen_funcs = set()

        # Update existing methods in Python code
        for methodname, j1, j2 in self.iter_methods(i1):
            seen_funcs.add(methodname)
            pre_lines = "\n".join(self.lines[j1 - 3 : j1])
            if "@apidiff.add" in pre_lines:
                pass
            elif self.method_is_known(classname, methodname):
                if "@apidiff.hide" in pre_lines:
                    pass  # continue as normal
                elif "@apidiff.change" in pre_lines:
                    continue
                old_line = self.lines[j1]
                new_line = self.get_method_def(classname, methodname)
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
            elif not methodname.startswith("_"):
                self.insert_line(
                    j1, f"    # FIXME: unknown method {classname}.{methodname}"
                )
                print(f"Unknown method {classname}.{methodname}")

        # Add missing methods for this class
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
        for methodname in self.get_required_method_names(classname):
            if methodname not in seen_funcs:
                lines.append("    # FIXME: new method to implement")
                lines.append(self.get_method_def(classname, methodname))
                lines.append("        raise NotImplementedError()\n")
        return lines


class IdlPatcherMixin:

    _map_funcname_idl2py = {"constructor": "__init__"}
    _map_funcname_py2idl = {v: k for k, v in _map_funcname_idl2py.items()}

    def __init__(self, idl):
        super().__init__()
        self.idl = idl

    def methodname_py2idl(self, name):
        return self._map_funcname_py2idl.get(name, name)

    def methodname_idl2py(self, name):
        return self._map_funcname_idl2py.get(name, name)

    def propname_py2idl(self, name):
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

    def propname_idl2py(self, name):
        name2 = ""
        for c in name:
            c_ = c.lower()
            if c_ != c:
                name2 += "_"
            name2 += c_
        return name2

    def class_is_known(self, classname):
        return classname in self.idl.classes

    def get_class_def(self, classname):
        cls = self.idl.classes[classname]
        # Make sure that GPUObjectBase comes last, for MRO
        bases = sorted(cls.bases or [], key=lambda n: n.count("GPUObjectBase"))
        # Cover some special cases
        bases = f"({', '.join(bases)})" if bases else ""
        if not bases and classname.lower().endswith("error"):
            bases = "(Exception)"
            if "memory" in classname:
                bases = "(MemoryError)"
        return f"class {classname}{bases}:"

    def get_method_def(self, classname, methodname):
        # Get the corresponding IDL line
        functions = self.idl.classes[classname].functions
        name_idl = self.methodname_py2idl(methodname)
        if methodname.endswith("_async") and name_idl not in functions:
            name_idl = self.methodname_py2idl(methodname.replace("_async", ""))
        idl_line = functions[name_idl]

        # Construct preamble
        preamble = "def " + to_python_name(methodname) + "("
        if "async" in methodname:
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

        # Get Python args, if one arg that is a dict, flatten dict to kwargs
        if len(argtypes) == 1 and argtypes[0].endswith(("Options", "Descriptor")):
            assert argtypes[0].startswith("GPU")
            arg_struct = self.idl.structs[argtypes[0][3:]]
            py_args = [field.py_arg() for field in arg_struct.values()]
            if py_args[0].startswith("label: str"):
                py_args[0] = 'label=""'
            py_args = ["self", "*"] + py_args
        else:
            py_args = ["self"] + argnames

        # Construct final def
        line = preamble + ", ".join(py_args) + "): pass\n"
        line = blacken(line, True).split("):")[0] + "):"
        return "    " + line

    def prop_is_known(self, classname, propname):
        propname_idl = self.propname_py2idl(propname)
        return propname_idl in self.idl.classes[classname].attributes

    def method_is_known(self, classname, methodname):
        functions = self.idl.classes[classname].functions
        name_idl = self.methodname_py2idl(methodname)
        if "_async" in methodname and name_idl not in functions:
            name_idl = self.methodname_py2idl(methodname.replace("_async", ""))
        return name_idl if name_idl in functions else None

    def get_class_names(self):
        return list(self.idl.classes.keys())

    def get_required_prop_names(self, classname):
        propnames_idl = self.idl.classes[classname].attributes.keys()
        return [self.propname_idl2py(x) for x in propnames_idl]

    def get_required_method_names(self, classname):
        methodnames_idl = self.idl.classes[classname].functions.keys()
        return [self.methodname_idl2py(x) for x in methodnames_idl]


class BaseApiPatcher(IdlPatcherMixin, AbstractApiPatcher):
    """A patcher to patch the base API (in base.py), using IDL as input."""


class IdlCommentInjector(IdlPatcherMixin, AbstractCommentInjector):
    """A patcher that injects signatures as defined in IDL, which can be useful
    to determine the types of arguments, etc.
    """

    def get_class_comment(self, classname):
        return None

    def get_prop_comment(self, classname, propname):
        if self.prop_is_known(classname, propname):
            propname_idl = self.propname_py2idl(propname)
            return "    # IDL: " + self.idl.classes[classname].attributes[propname_idl]

    def get_method_comment(self, classname, methodname):
        name_idl = self.method_is_known(classname, methodname)
        if name_idl:
            return "    # IDL: " + self.idl.classes[classname].functions[name_idl]


class BackendApiPatcher(AbstractApiPatcher):
    """A patcher to patch a backend API (e.g. rs.py), using the base API as input."""

    def __init__(self, base_api_code):
        super().__init__()

        p1 = Patcher(base_api_code)

        # Collect what's needed
        self.classes = classes = {}
        for classname, i1, i2 in p1.iter_classes():
            methods = {}
            for methodname, j1, j2 in p1.iter_methods(i1 + 1):
                pre_lines = "\n".join(p1.lines[j1 - 3 : j1])
                if "@apidiff.hide" in pre_lines:
                    continue  # method (currently) not part of our API
                body = "\n".join(p1.lines[j1 + 1 : j2 + 1])
                must_overload = "raise NotImplementedError()" in body
                methods[methodname] = p1.lines[j1], must_overload
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

    def get_method_def(self, classname, methodname):
        _, methods = self.classes[classname]
        line, _ = methods[methodname]
        return line

    def prop_is_known(self, classname, propname):
        return False

    def method_is_known(self, classname, methodname):
        _, methods = self.classes[classname]
        return methodname in methods

    def get_class_names(self):
        return list(self.classes.keys())

    def get_required_prop_names(self, classname):
        return []

    def get_required_method_names(self, classname):
        _, methods = self.classes[classname]
        return list(name for name in methods.keys() if methods[name][1])
