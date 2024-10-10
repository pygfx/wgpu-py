"""
The logic to generate/patch the base API from the WebGPU
spec (IDL), and the backend implementations from the base API.
"""

import ast
from collections import defaultdict
from functools import cache

from codegen.files import file_cache
from codegen.idlparser import Attribute, get_idl_parser
from codegen.utils import Patcher, format_code, print, to_camel_case, to_snake_case

# In wgpu-py, we make some args optional, that are not optional in the
# IDL. Reasons may be because it makes sense to be able to omit them,
# or because the WebGPU says its optional while IDL says it's not, or
# for backwards compatibility. These args have a default value of
# 'optional'  (which is just None) so we can recognise them. If IDL
# makes any of these args optional, their presence in this list is
# ignored.
ARGS_TO_MAKE_OPTIONAL = {
    ("compilation_hints", "compilation_hints"),  # idl actually has a default
    ("create_shader_module", "source_map"),
    ("begin_compute_pass", "timestamp_writes"),
    ("begin_render_pass", "timestamp_writes"),
    ("begin_render_pass", "depth_stencil_attachment"),
    ("begin_render_pass", "occlusion_query_set"),
    ("create_render_pipeline", "depth_stencil"),
    ("create_render_pipeline", "fragment"),
    ("create_render_pipeline_async", "depth_stencil"),
    ("create_render_pipeline_async", "fragment"),
    ("create_render_bundle_encoder", "depth_stencil_format"),
}


def patch_base_api(code):
    """Given the Python code, applies patches to make the code conform
    to the IDL.
    """
    idl = get_idl_parser()

    # Write __all__
    part1, found_all, part2 = code.partition("\n__all__ =")
    if found_all:
        part2 = part2.split("]", 1)[-1]
        line = "\n__all__ = ["
        line += ", ".join(f'"{name}"' for name in idl.classes.keys())
        line += "]"
        code = part1 + line + part2

    # Patch!
    for patcher in [CommentRemover(), BaseApiPatcher(), IdlCommentInjector()]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


def patch_backend_api(code):
    """Given the Python code, applies patches to make the code conform
    to the base API.
    """

    # Obtain the base API definition
    base_api_code = file_cache.read("_classes.py")

    # Patch!
    for patcher in [
        CommentRemover(),
        BackendApiPatcher(base_api_code),
        StructValidationChecker(),
    ]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


class CommentRemover(Patcher):
    """A patcher that removes comments that we add in other parsers,
    to prevent accumulating comments.
    """

    triggers = "# IDL:", "# FIXME: unknown api", "# FIXME: missing check_struct"

    def apply(self, code):
        self._init(code)
        for line, i in self.iter_lines():
            if line.lstrip().startswith(self.triggers):
                self.remove_line(i)


class AbstractCommentInjector(Patcher):
    """A base patcher that can insert  helpful comments in front of
    properties, methods, and classes. It does not mark any as new or unknown,
    since that is the task of the API patchers.

    Also moves decorators just above the def. Doing this here in a
    post-processing step means we don't have to worry about decorators
    in the other patchers, keeping them simpler.
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
            self._move_decorator_below_comments(j1)

    def patch_methods(self, classname, i1, i2):
        for methodname, j1, j2 in self.iter_methods(i1):
            comment = self.get_method_comment(classname, methodname)
            if comment:
                self.insert_line(j1, comment)
            self._move_decorator_below_comments(j1)

    def _move_decorator_below_comments(self, i_def):
        for i in range(i_def - 3, i_def):
            line = self.lines[i]
            if line.lstrip().startswith("@"):
                self.remove_line(i)
                self.insert_line(i_def, line)


class AbstractApiPatcher(Patcher):
    """The base patcher to update a wgpu API.

    This code is generalized, so it can be used both to generate the base API
    and the backends (implementations).

    The idea is to walk over all classes, patch it if necessary, then
    walk over each of its properties and methods to patch these too.
    """

    def apply(self, code):
        self._init(code)
        self._counts = {"classes": 0, "methods": 0, "properties": 0}
        self.patch_classes()
        stats = ", ".join(f"{self._counts[key]} {key}" for key in self._counts)
        print("Validated " + stats)

    def patch_classes(self):
        seen_classes = set()

        # Update existing classes in the Python code
        for classname, i1, i2 in self.iter_classes():
            seen_classes.add(classname)
            self._apidiffs = set()
            if self.class_is_known(classname):
                old_line = self.lines[i1]
                new_line = self.get_class_def(classname)
                if old_line != new_line:
                    fixme_line = "# FIXME: was " + old_line.split("class ", 1)[-1]
                    self.replace_line(i1, f"{fixme_line}\n{new_line}")
                self.patch_properties(classname, i1 + 1, i2)
                self.patch_methods(classname, i1 + 1, i2)
            else:
                msg = f"unknown api: class {classname}"
                self.insert_line(i1, "# FIXME: " + msg)
                print("Warning: " + msg)
            if self._apidiffs:
                print(f"Diffs for {classname}:", ", ".join(sorted(self._apidiffs)))

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
            self.insert_line(i2 + 1, "\n".join(lines))

        self._counts["classes"] += len(seen_classes)

    def patch_properties(self, classname, i1, i2):
        seen_props = set()

        # Update existing properties in Python code
        for propname, j1, j2 in self.iter_properties(i1):
            seen_props.add(propname)
            pre_lines = "\n".join(self.lines[j1 - 3 : j1])
            self._apidiffs_from_lines(pre_lines, propname)
            if self.prop_is_known(classname, propname):
                if "@apidiff.add" in pre_lines:
                    print(f"ERROR: apidiff.add for known {classname}.{propname}")
                elif "@apidiff.hide" in pre_lines:
                    pass  # continue as normal
                old_line = self.lines[j1]
                new_line = self.get_property_def(classname, propname)
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
            elif "@apidiff.add" in pre_lines:
                pass
            else:
                msg = f"unknown api: prop {classname}.{propname}"
                self.insert_line(j1, "    # FIXME: " + msg)
                print("Warning: " + msg)

        # Add missing properties for this class
        lines = self.get_missing_properties(classname, seen_props)
        if lines:
            self.insert_line(i2 + 1, "\n".join(lines))

        self._counts["properties"] += len(seen_props)

    def patch_methods(self, classname, i1, i2):
        seen_funcs = set()

        # Update existing methods in Python code
        for methodname, j1, j2 in self.iter_methods(i1):
            seen_funcs.add(methodname)
            pre_lines = "\n".join(self.lines[j1 - 3 : j1])
            self._apidiffs_from_lines(pre_lines, methodname)
            if self.method_is_known(classname, methodname):
                if "@apidiff.add" in pre_lines:
                    print(f"ERROR: apidiff.add for known {classname}.{methodname}")
                elif "@apidiff.hide" in pre_lines:
                    pass  # continue as normal
                elif "@apidiff.change" in pre_lines:
                    continue
                old_line = self.lines[j1]
                new_line = self.get_method_def(classname, methodname)
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
            elif "@apidiff.add" in pre_lines:
                pass
            elif methodname.startswith("_"):
                pass
            else:
                msg = f"unknown api: method {classname}.{methodname}"
                self.insert_line(j1, "    # FIXME: " + msg)
                print("Warning: " + msg)

        # Add missing methods for this class
        lines = self.get_missing_methods(classname, seen_funcs)
        if lines:
            self.insert_line(i2 + 1, "\n".join(lines))

        self._counts["methods"] += len(seen_funcs)

    def get_missing_properties(self, classname, seen_props):
        lines = []
        for propname in self.get_required_prop_names(classname):
            if propname not in seen_props:
                lines.append("    # FIXME: new prop to implement")
                lines.append("    @property")
                lines.append(self.get_property_def(classname, propname))
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

    def _apidiffs_from_lines(self, text, what):
        diffs = [x.replace("(", " ").split()[0] for x in text.split("@apidiff.")[1:]]
        if diffs:
            self._apidiffs.add(f"{'/'.join(diffs)} {what}")


class IdlPatcherMixin:
    def __init__(self):
        super().__init__()
        self.idl = get_idl_parser()
        self.detect_async_props_and_methods()

    def detect_async_props_and_methods(self):
        self.async_idl_names = async_idl_names = {}  # (sync-name, async-name)

        for classname, interface in self.idl.classes.items():
            for namedict in [interface.attributes, interface.functions]:
                for name_idl, idl_line in namedict.items():
                    idl_result = idl_line.split(name_idl)[0]
                    if "Promise" in idl_result:
                        # We found an async property or method.
                        name_idl_base = name_idl
                        if name_idl.endswith("Async"):
                            name_idl_base = name_idl[:-5]
                        key = classname, name_idl_base
                        # Now we determine the kind
                        if name_idl_base != name_idl and name_idl_base in namedict:
                            # Has both
                            async_idl_names[key] = name_idl_base, name_idl
                        else:
                            # Only has async
                            async_idl_names[key] = None, name_idl

    def get_idl_name_variants(self, classname, base_name):
        """Returns the names of an idl prop/method for its sync and async variant.
        Either can be None.
        """
        # Must be a base name, without the suffix
        assert not base_name.lower().endswith(("sync", "async"))

        key = classname, base_name
        default = base_name, None
        return self.async_idl_names.get(key, default)

    def name2idl(self, classname, name_py):
        """Map a python propname/methodname to the idl variant.
        Take async into account.
        """
        if name_py == "__init__":
            return "constructor"

        # Get idl base name
        if name_py.endswith(("_sync", "_async")):
            name_idl_base = to_camel_case(name_py.rsplit("_", 1)[0])
        else:
            name_idl_base = to_camel_case(name_py)

        # Get idl variant names
        idl_sync, idl_async = self.get_idl_name_variants(classname, name_idl_base)

        # Triage
        if idl_sync and idl_async:
            if name_py.endswith("_async"):
                return idl_async
            elif name_py.endswith("_sync"):
                return name_idl_base + "InvalidVariant"
            else:
                return idl_sync
        elif idl_async:
            if name_py.endswith("_async"):
                return idl_async
            elif name_py.endswith("_sync"):
                return idl_async
            else:
                return name_idl_base + "InvalidVariant"
        else:  # idl_sync only
            if name_py.endswith("_async"):
                return name_idl_base + "InvalidVariant"
            elif name_py.endswith("_sync"):
                return name_idl_base + "InvalidVariant"
            else:
                return idl_sync

    def name2py_names(self, classname, name_idl):
        """Map an idl propname/methodname to the python variants.
        Take async into account. Returns a list with one or two names;
        for async props/methods Python has the sync and the async variant.
        """

        if name_idl == "constructor":
            return ["__init__"]

        # Get idl base name
        name_idl_base = name_idl
        if name_idl.endswith("Async"):
            name_idl_base = name_idl[:-5]
        name_py_base = to_snake_case(name_idl_base)

        # Get idl variant names
        idl_sync, idl_async = self.get_idl_name_variants(classname, name_idl_base)

        if idl_sync and idl_async:
            return [to_snake_case(idl_sync), name_py_base + "_async"]
        elif idl_async:
            return [name_py_base + "_sync", name_py_base + "_async"]
        else:
            assert idl_sync == name_idl_base
            return [name_py_base]

    def class_is_known(self, classname):
        return classname in self.idl.classes

    def get_class_def(self, classname):
        cls = self.idl.classes[classname]
        # Make sure that GPUObjectBase comes last, for MRO
        ignore = "Event", "EventTarget", "DOMException"
        bases = sorted(cls.bases or [], key=lambda n: n.count("GPUObjectBase"))
        bases = [b for b in bases if b not in ignore]
        # Cover some special cases
        if classname.lower().endswith("error"):
            if "memory" in classname.lower():
                bases.append("MemoryError")
            elif not bases:
                bases.append("Exception")

        bases = "" if not bases else f"({', '.join(bases)})"
        return f"class {classname}{bases}:"

    def get_property_def(self, classname, propname):
        attributes = self.idl.classes[classname].attributes
        name_idl = self.name2idl(classname, propname)
        assert name_idl in attributes

        line = "def " + to_snake_case(propname) + "(self):"
        if propname.endswith("_async"):
            line = "async " + line
        return "    " + line

    def get_method_def(self, classname, methodname):
        functions = self.idl.classes[classname].functions
        name_idl = self.name2idl(classname, methodname)
        assert name_idl in functions

        # Construct preamble
        preamble = "def " + to_snake_case(methodname) + "("
        if methodname.endswith("_async"):
            preamble = "async " + preamble

        # Get arg names and types
        idl_line = functions[name_idl]
        args = idl_line.split("(", 1)[1].split(")", 1)[0].split(",")
        args = [Attribute(arg) for arg in args if arg.strip()]

        # If one arg that is a dict, flatten dict to kwargs
        if len(args) == 1 and args[0].typename.endswith(
            ("Options", "Descriptor", "Configuration")
        ):
            assert args[0].typename.startswith("GPU")
            des_is_optional = bool(args[0].default)
            attributes = self.idl.structs[args[0].typename[3:]].values()
            py_args = [
                self._arg_from_attribute(methodname, attr, des_is_optional)
                for attr in attributes
            ]
            if py_args[0].startswith("label: str"):
                py_args[0] = 'label: str=""'
            py_args = ["self", "*", *py_args]
        else:
            py_args = [self._arg_from_attribute(methodname, attr) for attr in args]
            py_args = ["self", *py_args]

            # IDL has some signatures that cannot work in Python. This may be a bug in idl
            known_bugged_methods = {"GPUPipelineError.__init__"}
            remove_default = False
            for i in reversed(range(len(py_args))):
                arg = py_args[i]
                if "=" in arg:
                    if remove_default:
                        py_args[i] = arg.split("=")[0]
                        assert f"{classname}.{methodname}" in known_bugged_methods
                else:
                    remove_default = True

        # Construct final def
        line = preamble + ", ".join(py_args) + "): pass\n"
        line = format_code(line, True).split("):")[0] + "):"
        return "    " + line

    def _arg_from_attribute(self, methodname, attribute, force_optional=False):
        name = to_snake_case(attribute.name)
        optional_in_py = (methodname, name) in ARGS_TO_MAKE_OPTIONAL
        d = attribute.default
        t = self.idl.resolve_type(attribute.typename)
        result = name
        if (force_optional or optional_in_py) and not d:
            d = "optional"
        if t:
            # If default is None, the type won't match, so we need to mark it optional
            if d == "None":
                result += f": Optional[{t}]"
            else:
                result += f": {t}"
        if d:
            d = {"false": "False", "true": "True"}.get(d, d)
            result += f"={d}"
        return result

    def prop_is_known(self, classname, propname):
        attributes = self.idl.classes[classname].attributes
        propname_idl = self.name2idl(classname, propname)
        return propname_idl if propname_idl in attributes else None

    def method_is_known(self, classname, methodname):
        functions = self.idl.classes[classname].functions
        methodname_idl = self.name2idl(classname, methodname)
        return methodname_idl if methodname_idl in functions else None

    def get_class_names(self):
        return list(self.idl.classes.keys())

    def get_required_prop_names(self, classname):
        attributes = self.idl.classes[classname].attributes
        names = []
        for name_idl in attributes.keys():
            names.extend(self.name2py_names(classname, name_idl))
        return names

    def get_required_method_names(self, classname):
        functions = self.idl.classes[classname].functions
        names = []
        for name_idl in functions.keys():
            names.extend(self.name2py_names(classname, name_idl))
        return names


class BaseApiPatcher(IdlPatcherMixin, AbstractApiPatcher):
    """A patcher to patch the base API (in _classes.py), using IDL as input."""


class IdlCommentInjector(IdlPatcherMixin, AbstractCommentInjector):
    """A patcher that injects signatures as defined in IDL, which can be useful
    to determine the types of arguments, etc.
    """

    def get_class_comment(self, classname):
        return None

    def get_prop_comment(self, classname, propname):
        attributes = self.idl.classes[classname].attributes
        name_idl = self.prop_is_known(classname, propname)
        if name_idl:
            return "    # IDL: " + attributes[name_idl]

    def get_method_comment(self, classname, methodname):
        functions = self.idl.classes[classname].functions
        name_idl = self.method_is_known(classname, methodname)
        if name_idl:
            idl_line = functions[name_idl]

            args = idl_line.split("(", 1)[1].split(")", 1)[0].split(",")
            args = [Attribute(arg) for arg in args if arg.strip()]

            # If one arg that is a dict, flatten dict to kwargs
            if len(args) == 1 and args[0].typename.endswith(
                ("Options", "Descriptor", "Configuration")
            ):
                assert args[0].typename.startswith("GPU")
                attributes = self.idl.structs[args[0].typename[3:]].values()
                idl_line += " -> " + ", ".join(attr.line for attr in attributes)

            return "    # IDL: " + idl_line


class BackendApiPatcher(AbstractApiPatcher):
    """A patcher to patch a backend API, using the base API as input."""

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

        if "):" not in line:
            return line.replace(":", f"(classes.{classname}):")
        else:
            i = line.find("(")
            bases = line[i:].strip("():").replace(",", " ").split()
            bases = [b for b in bases if b.startswith("GPU")]
            bases.insert(0, f"classes.{classname}")
            return line[:i] + "(" + ", ".join(bases) + "):"

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


class StructValidationChecker(Patcher):
    """Checks that all structs are validated in the methods that have incoming structs."""

    def apply(self, code):
        self._init(code)

        idl = get_idl_parser()
        all_structs = set()
        ignore_structs = {"Extent3D", "Origin3D"}

        structure_checks = self.get_structure_checks()

        for classname, i1, i2 in self.iter_classes():
            if classname not in idl.classes:
                continue

            # For each method ...
            for methodname, j1, j2 in self.iter_methods(i1 + 1):
                code = "\n".join(self.lines[j1 : j2 + 1])
                # Get signature and cut it up in words
                sig_words = code.partition("(")[2].split("):")[0]
                for c in "][(),\"'":
                    sig_words = sig_words.replace(c, " ")
                # Collect incoming structs from signature
                method_structs = set()
                for word in sig_words.split():
                    if word.startswith("structs."):
                        structname = word.partition(".")[2]
                        method_structs.update(self._get_sub_structs(idl, structname))
                all_structs.update(method_structs)
                # Collect structs being checked
                checked = structure_checks[classname, methodname]

                # Test that a matching check is done
                unchecked = method_structs.difference(checked)
                unchecked = list(sorted(unchecked.difference(ignore_structs)))
                # Must we check, or does this method defer to another
                defer_func_name = "_" + methodname
                defer_line_starts = (
                    f"return self.{defer_func_name[:-7]}",
                    f"awaitable = self.{defer_func_name[:-7]}",
                )
                this_method_defers = any(
                    line.strip().startswith(defer_line_starts)
                    for line in code.splitlines()
                )
                if not this_method_defers and unchecked:
                    msg = f"missing check_struct in {methodname}: {unchecked}"
                    self.insert_line(j1, f"# FIXME: {msg}")
                    print(f"ERROR: {msg}")

        # Test that we did find structs. In case our detection fails for
        # some reason, this would probably catch that.
        assert len(all_structs) > 10

    def _get_sub_structs(self, idl, structname):
        structnames = {structname}
        for attribute in idl.structs[structname].values():
            structname2 = attribute.typename[3:]  # remove "GPU"
            if structname2 in idl.structs:
                structnames.update(self._get_sub_structs(idl, structname2))
        return structnames

    @staticmethod
    def get_structure_checks():
        module = ast.parse(file_cache.read("backends/wgpu_native/_api.py"))
        # We only care about top-level classes and their top-level methods.
        top_level_methods = {
            # (class_name, method_name) -> method_ast
            (class_ast.name, method_ast.name): method_ast
            for class_ast in module.body
            if isinstance(class_ast, ast.ClassDef)
            for method_ast in class_ast.body
            if isinstance(method_ast, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        # (class_name, method_name) -> list of helper methods
        method_helper_calls = defaultdict(list)
        # (class_name, method_name) -> list of structures checked
        structure_checks = defaultdict(list)

        for key, method_ast in top_level_methods.items():
            for node in ast.walk(method_ast):
                if isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                    if name.startswith("self._"):
                        method_helper_calls[key].append(name[5:])
                    if name == "check_struct":
                        if isinstance(node.args[0], ast.Constant):
                            struct_name = node.args[0].value
                            structure_checks[key].append(struct_name)

        @cache
        def get_function_checks(class_name, method_name):
            result = set(structure_checks[class_name, method_name])
            for helper_method_name in method_helper_calls[class_name, method_name]:
                result.update(get_function_checks(class_name, helper_method_name))
            return sorted(result)

        return {key: get_function_checks(*key) for key in top_level_methods.keys()}
