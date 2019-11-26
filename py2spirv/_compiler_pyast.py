import struct
import inspect

from ._module import SpirVModule
from ._generator_base import BaseSpirVGenerator, STORAGE_CLASSES
from . import commonast
from . import _spirv_constants as cc


# DEPRECATED. WILL PROBABLY BE REMOVED SOON.


def pythonast2spirv(func):
    raise NotImplementedError()


class Ast2SpirVGenerator(BaseSpirVGenerator):
    """ WIP Python 2 SpirV Compiler that parses Python AST to generate
    SpirV code. A downside of this approach is that to get the AST,
    the Python source must be available.
    """

    def _prepare(self):
        py_code = inspect.getsource(self._py_func)
        self._py_ast = commonast.parse(py_code)

        if not (
            len(self._py_ast.body_nodes) == 1
            and isinstance(self._py_ast.body_nodes[0], commonast.FunctionDef)
        ):
            raise ValueError("AST does not resolve to a function node.")

    def _generate(self):
        # Now walk the AST!
        self.scope_stack.append({})

        for node in self._py_ast.body_nodes[0].body_nodes:
            self.parse(node)

    def parse(self, node):
        nodeType = node.__class__.__name__
        parse_func = getattr(self, "parse_" + nodeType, None)
        if parse_func:
            return parse_func(node)
        else:
            raise Exception("Cannot parse %s-nodes yet" % nodeType)

    def parse_FunctionDef(self, node):

        assert node.name == "main"

        self.scope_stack.append({})
        result_type_id = self.get_type_id("void")
        function_id = self._entry_point_id  # self.create_id(result_type_id)

        # Generate the function type -> specify args and return value
        function_type_id = self.create_id(result_type_id)
        self.gen_instruction(
            "types", cc.OpTypeFunction, function_type_id, result_type_id, *[]
        )  # can haz arg ids

        # todo: also generate instructions in function_defs section

        # Generate function
        function_control = 0  # can specify whether it should inline, etc.
        self.gen_instruction(
            "functions",
            cc.OpFunction,
            result_type_id,
            function_id,
            function_control,
            function_type_id,
        )

        # A function must begin with a label
        self.gen_instruction("functions", cc.OpLabel, self.create_id("label"))

        for sub in node.body_nodes:
            self.parse(sub)

        self.gen_instruction("functions", cc.OpReturn)
        self.gen_instruction("functions", cc.OpFunctionEnd)
        self.scope_stack.pop(-1)

    def parse_Assign(self, node):
        # scope = self.scope_stack[-1]
        assert len(node.target_nodes) == 1
        varname = node.target_nodes[0].name

        if len(self.scope_stack) == 1:

            if isinstance(node.value_node, commonast.Call):
                funcname = node.value_node.func_node.name
                argnames = []
                for arg in node.value_node.arg_nodes:
                    if isinstance(arg, commonast.Num):
                        argnames.append(arg.value)
                    elif isinstance(arg, commonast.Name):
                        argnames.append(arg.name)
                    else:
                        raise NotImplementedError()
                if funcname in ("input", "output"):
                    assert len(argnames) == 1
                    vartype = argnames[0]
                    self.create_variable(vartype, varname, STORAGE_CLASSES[funcname])
                else:
                    # scope[varname] = ??
                    raise NotImplementedError()

            else:
                # A constant, I guess
                if isinstance(node.value_node, commonast.List):
                    list_vartypes = []
                    for subnode in node.value_node.element_nodes:
                        assert isinstance(subnode, commonast.Call)
                        list_vartypes.append(subnode.func_node.name)
                    list_vartypes2 = set(list_vartypes)
                    if len(list_vartypes2) != 1:
                        raise TypeError("Lists must have uniform element types")
                    vartype = f"array_{len(list_vartypes)}_{list_vartypes[0]}"
                else:
                    raise NotImplementedError()
                self.create_variable(vartype, varname, STORAGE_CLASSES["constant"])

        else:
            root = self.scope_stack[0]
            assert varname in root, "can only use global vars yet"
            _, target_id, _, _ = self.get_variable_info(varname)

            result_id = self.parse(node.value_node)
            assert result_id
            self.gen_func_instruction(cc.OpStore, target_id, result_id, 0)

    def parse_Call(self, node):
        funcname = node.func_node.name
        arg_ids = [self.parse(arg) for arg in node.arg_nodes]
        if funcname in ("vec2", "vec3", "vec4"):
            composite_ids = []
            for id in arg_ids:
                the_type = self.get_type_from_id(id)
                type_name = (
                    the_type.__name__
                )  # self.get_type_id(type_id)  # todo: this is weird
                if type_name == "float":
                    composite_ids.append(id)
                elif type_name in ("vec2", "vec3", "vec4"):
                    for i in range(int(type_name[3:])):
                        type_id = self.get_type_id("float")
                        comp_id = self.create_id(type_id)
                        self.gen_func_instruction(
                            cc.OpCompositeExtract, type_id, comp_id, id, i
                        )
                        composite_ids.append(comp_id)
                else:
                    raise TypeError(f"Cannot convert create vec4 from {type}")
            if len(composite_ids) != int(funcname[3:]):
                raise TypeError(
                    f"{funcname} did not expect {len(composite_ids)} elements"
                )
            type_id = self.get_type_id(funcname)
            result_id = self.create_id(type_id)
            self.gen_func_instruction(
                cc.OpCompositeConstruct, type_id, result_id, *composite_ids
            )
            return result_id
        else:
            raise NotImplementedError()

    def parse_Name(self, node):
        name, id, type, type_id = self.get_variable_info(node.name)

        # todo: only load when the name is a pointer, and only once per func
        load_id = self.create_id(type_id)
        self.gen_func_instruction(cc.OpLoad, type_id, load_id, id)
        return load_id

    def parse_Num(self, node):
        # todo: re-use constants
        if isinstance(node.value, int):
            type_id = self.get_type_id("int")
            result_id = self.create_id(type_id)
            self.gen_instruction(
                "types",
                cc.OpConstant,
                type_id,
                result_id,
                struct.pack("<I", node.value),
            )
        elif isinstance(node.value, float):
            type_id = self.get_type_id("float")
            result_id = self.create_id(type_id)
            self.gen_instruction(
                "types",
                cc.OpConstant,
                type_id,
                result_id,
                struct.pack("<f", node.value),
            )
        return result_id

    def parse_Subscript(self, node):
        if not isinstance(node.value_node, commonast.Name):
            raise TypeError("Can only slice into direct variables.")
        if not isinstance(node.slice_node, commonast.Index):
            raise TypeError("Only singleton indices allowed.")

        value_node = node.value_node
        index_node = node.slice_node.value_node

        self.parse(value_node)
        self.parse(index_node)
        # self.gen_func_instruction()

        type_id = self.get_type_id("vec2")
        result_id = self.create_id(type_id)

        return result_id
