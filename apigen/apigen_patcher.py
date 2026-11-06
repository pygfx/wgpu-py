import ast
import os
import tempfile
from collections import defaultdict
from functools import cache

from apigen_utils import FormatError, format_code, read_file


class Patcher:
    """Class to help patch a Python module. Supports iterating (over
    lines, classes, properties, methods), and applying diffs (replace,
    remove, insert).
    """

    def __init__(self, code=None):
        self._init(code)

    def _init(self, code):
        """Subclasses can call this to reset the patcher."""
        self.lines = []
        self._diffs = {}
        self._classes = {}
        if code:
            self.lines = format_code(code, True).splitlines()  # inf line length

    def remove_line(self, i):
        """Remove the line at the given position. There must not have been
        an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "remove"

    def insert_line(self, i, line):
        """Insert a new line at the given position. It's ok if there
        has already been an insertion on line i, but there must not have been
        any other actions.
        """
        if i in self._diffs and self._diffs[i][1] == "insert":
            cur_line = self._diffs[i][2]
            self._diffs[i] = i, "insert", cur_line + "\n" + line
        else:
            assert i not in self._diffs, f"Line {i} already has a diff"
            self._diffs[i] = i, "insert", line

    def replace_line(self, i, line):
        """Replace the line at the given position with another line.
        There must not have been an action on line i.
        """
        assert i not in self._diffs, f"Line {i} already has a diff"
        self._diffs[i] = i, "replace", line

    def dumps(self, format=True):
        """Return the patched result as a string."""
        lines = self.lines.copy()
        # Apply diff
        diffs = sorted(self._diffs.values())
        for diff in reversed(diffs):
            if diff[1] == "remove":
                lines.pop(diff[0])
            elif diff[1] == "insert":
                lines.insert(diff[0], diff[2])
            elif diff[1] == "replace":
                lines[diff[0]] = diff[2]
            else:  # pragma: no cover
                raise ValueError(f"Unknown diff: {diff}")
        # Format
        text = "\n".join(lines)
        if format:
            try:
                text = format_code(text)
            except FormatError as err:  # pragma: no cover
                # If you get this error, it really helps to load the code
                # in an IDE to see where the error is. Let's help with that ...
                filename = os.path.join(tempfile.gettempdir(), "wgpu_patcher_fail.py")
                with open(filename, "wb") as f:
                    f.write(text.encode())
                err = str(err)
                err = err if len(err) < 78 else err[:77] + "…"
                raise RuntimeError(
                    f"It appears that the patcher has generated invalid Python:"
                    f"\n\n    {err}\n\n"
                    f'Wrote the generated (but unformatted) code to:\n\n  "{filename}"'
                ) from None

        return text

    def iter_lines(self, start_line=0):
        """Generator to iterate over the lines.
        Each iteration yields (line, linenr)
        """
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            yield line, i

    def iter_classes(self, start_line=0):
        """Generator to iterate over the classes.
        Each iteration yields (classname, linenr_start, linenr_end),
        where linenr_end is the last line of code.
        """
        current_class = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_class and sline:
                if sline.startswith("    "):
                    current_class[2] = i
                else:  # code has less indentation -> something new
                    yield current_class
                    current_class = None
            if line.startswith("class "):
                name = line.split(":")[0].split("(")[0].split()[-1]
                current_class = [name, i, i]
        if current_class:
            yield current_class

    def iter_properties(self, start_line=0):
        """Generator to iterate over the properties.
        Each iteration yields (classname, linenr_first, linenr_last),
        where linenr_first is the line that startswith `def`,
        and linenr_last is the last line of code.
        """
        return self._iter_props_and_methods(start_line, True)

    def iter_methods(self, start_line=0):
        """Generator to iterate over the methods.
        Each iteration yields (classname, linenr_first, linenr_last)
        where linenr_first is the line that startswith `def`,
        and linenr_last is the last line of code.
        """
        return self._iter_props_and_methods(start_line, False)

    def _iter_props_and_methods(self, start_line, find_props):
        prop_mark = None
        current_def = None
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            sline = line.rstrip()
            if current_def and sline:
                if sline.startswith("        "):
                    current_def[2] = i
                else:
                    yield current_def
                    current_def = None
            if sline and not sline.startswith("    "):
                break  # exit class
            if line.startswith(("    def ", "    async def ")):
                name = line.split("(")[0].split()[-1]
                if prop_mark and find_props:
                    current_def = [name, i, i]
                elif not prop_mark and not find_props:
                    current_def = [name, i, i]
            if line.startswith("    @property"):
                prop_mark = i
            elif sline and not sline.lstrip().startswith("#"):
                prop_mark = None

        if current_def:
            yield current_def


class CommentRemover(Patcher):
    """A patcher that removes comments that we add in other parsers,
    to prevent accumulating comments.
    """

    def __init__(
        self, triggers=("# FIXME: unknown api", "# FIXME: missing check_struct")
    ):
        # TODO: add "# IDL:" temporarily later to remove legacy stuff
        super().__init__()
        self.triggers = triggers

    def apply(self, code):
        self._init(code)
        for line, i in self.iter_lines():
            if line.lstrip().startswith(self.triggers):
                self.remove_line(i)


class ApiPatcher(Patcher):
    """The patcher to update an API (classes and their props and methods).

    The idea is to walk over all classes, patch it if necessary, then
    walk over each of its properties and methods to patch these too.
    """

    known_external_classes = {"GPUPromise"}

    def __init__(self, api_dict, explode_single_args_that_are_descriptors: bool):
        super().__init__()
        self.api_dict = api_dict
        self.explode_single_args_that_are_descriptors = (
            explode_single_args_that_are_descriptors
        )

    def apply(self, code):
        self._init(code)
        self._counts = {"classes": 0, "methods": 0, "properties": 0}
        self.patch_classes()
        stats = ", ".join(f"{self._counts[key]} {key}" for key in self._counts)
        print("Patched " + stats)

    def patch_classes(self):
        seen_classes = set()

        # Update existing classes in the Python code
        for classname, i1, i2 in self.iter_classes():
            seen_classes.add(classname)
            self._apidiffs = set()
            if classname in self.known_external_classes:
                pass
            elif self.class_is_known(classname):
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
            if self.prop_is_known(classname, propname):
                old_line = self.lines[j1]
                new_line = self.get_property_def(classname, propname)
                if old_line != new_line.split("@property\n")[1]:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
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
            if self.method_is_known(classname, methodname):
                old_line = self.lines[j1]
                new_line = self.get_method_def(classname, methodname)
                if old_line != new_line:
                    fixme_line = "    # FIXME: was " + old_line.split("def ", 1)[-1]
                    lines = [fixme_line, new_line]
                    self.replace_line(j1, "\n".join(lines))
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
        for propname in self.get_prop_names(classname):
            if propname not in seen_props:
                lines.append("    # FIXME: new prop to implement")
                lines.append(self.get_property_def(classname, propname))
                lines.append("        raise NotImplementedError()")
                lines.append("")
        return lines

    def get_missing_methods(self, classname, seen_funcs):
        lines = []
        for methodname in self.get_method_names(classname):
            if methodname not in seen_funcs:
                lines.append("    # FIXME: new method to implement")
                lines.append(self.get_method_def(classname, methodname))
                lines.append("        raise NotImplementedError()\n")
        return lines

    def get_class_def(self, class_name):
        class_dict = self.api_dict["classes"][class_name]
        bases = class_dict.get("bases", [])
        bases_str = ""
        if bases:
            bases_str = "(" + ", ".join(bases) + ")"
        return f"class {class_name}{bases_str}:"

    def get_property_def(self, class_name, prop_name) -> str:
        prop_dict = self.api_dict["classes"][class_name]["properties"][prop_name]
        return f"    @property\n    def {prop_name}(self) -> {prop_dict['type']}:"

    def get_method_def(self, class_name, method_name) -> str:
        method_dict = self.api_dict["classes"][class_name]["methods"][method_name]
        args = ["self"]
        args_dict = method_dict["args"]
        # Maybe use descriptor field as args
        if (
            self.explode_single_args_that_are_descriptors
            and len(method_dict["args"]) == 1
        ):
            arg_type_name = list(method_dict["args"].values())[0]["type"]
            if arg_type_name.startswith("structs."):
                if arg_type_name.endswith(" | None"):
                    arg_type_name = arg_type_name[:-7]
                if arg_type_name.endswith("Struct"):
                    arg_type_name = arg_type_name[:-6]
                args.append("*")
                args_dict = self.api_dict["structs"][arg_type_name[8:]]["fields"]
        # Maybe make the args non-positional
        if "*" not in args:
            need_star = False
            previous_where_all_required = True
            for arg_name, arg_dict in args_dict.items():
                if arg_dict["default"]:
                    previous_where_all_required = False
                elif not previous_where_all_required:
                    need_star = True
            if need_star:
                args.append("*")
        # Build signature
        for arg_name, arg_dict in args_dict.items():
            typ = arg_dict["type"]
            arg = f"{arg_name}: {typ}"
            if arg_dict["default"]:
                arg += f" = {arg_dict['default']}"
            args.append(arg)
        sig = ", ".join(args)
        return f"    def {method_name}({sig}) -> {method_dict['returns']['type']}:"

    def class_is_known(self, class_name):
        return class_name in self.api_dict["classes"]

    def prop_is_known(self, class_name, prop_name):
        return prop_name in self.api_dict["classes"][class_name]["properties"]

    def method_is_known(self, class_name, method_name):
        return method_name in self.api_dict["classes"][class_name]["methods"]

    def get_class_names(self):
        return list(self.api_dict["classes"].keys())

    def get_prop_names(self, class_name):
        return list(self.api_dict["classes"][class_name].get("properties", {}).keys())

    def get_method_names(self, class_name):
        return list(self.api_dict["classes"][class_name].get("methods", {}).keys())


class StructValidationChecker(Patcher):
    """Checks that all structs are validated in the methods that have incoming structs."""

    def __init__(self, api_dict):
        super().__init__()
        self.api_dict = api_dict

    def apply(self, code):
        self._init(code)

        all_structs = set()
        ignore_structs = {
            "Extent3D",
            "Origin3D",
        }  # TODO: ignore here or convert earlier?

        structure_checks = self._get_structure_checks()

        for classname, i1, i2 in self.iter_classes():
            if classname not in self.api_dict["classes"]:
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
                        method_structs.update(self._get_sub_structs(structname))
                all_structs.update(method_structs)
                # Collect structs being checked
                try:
                    checked = structure_checks[classname, methodname]
                except KeyError:
                    checked = set()
                    print(
                        f"Could not validate check_struct for {classname}.{methodname}; need another codegen run."
                    )

                # Test that a matching check is done
                unchecked = method_structs.difference(checked)
                unchecked = list(sorted(unchecked.difference(ignore_structs)))
                # Must we check, or does this method defer to another
                defer_func_name = "_" + methodname
                defer_line_starts = (
                    f"return self.{defer_func_name[:-7]}",
                    f"promise = self.{defer_func_name[:-7]}",
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

    def _get_sub_structs(self, structname):
        # todo: how to deal with these suffixes?
        # if structname.endswith("Struct"):
        #     structname = structname[:-6]
        structnames = {structname}
        for field_name, field_dict in self.api_dict.structs[structname].items():
            type_name = field_dict["type"]
            if type_name in self.api_dict["structs"]:
                structnames.update(self._get_sub_structs(type_name))
        return structnames

    @staticmethod
    def _get_structure_checks():
        """
        Returns a map
            (class_name, method_name) -> <list of structure names>
        mapping each top-level method in _api.py to the calls to check_struct made by
        that method or by any helper methods called by that method.

        For now, the helper function must be methods within the same class. This code
        does not yet deal with global functions or with methods in superclasses.
        """
        # TODO: abstract the file, we want to do this for each backend, I suppose
        module = ast.parse(read_file("wgpu", "backends", "wgpu_native", "_api.py"))
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
                        assert isinstance(node.args[0], ast.Constant)
                        struct_name = node.args[0].value
                        assert isinstance(struct_name, str)
                        structure_checks[key].append(struct_name)

        @cache
        def get_function_checks(class_name, method_name):
            result = set(structure_checks[class_name, method_name])
            for helper_method_name in method_helper_calls[class_name, method_name]:
                result.update(get_function_checks(class_name, helper_method_name))
            return sorted(result)

        return {key: get_function_checks(*key) for key in top_level_methods.keys()}
