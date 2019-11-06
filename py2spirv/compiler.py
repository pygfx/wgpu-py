"""
A compiler to translate Python code to SpirV. A compiler can be
considered to exists of two parts: a parser to go from one code (Python)
to an AST, and a generator to transform that AST into another code
(SpirV). In this case the parsing is done using Python's builtin ast
module, so the main part of this module comprises the SpirV generation.

References:
* https://www.khronos.org/registry/spir-v/

"""

import os
import io
import ast
import struct
import inspect
import tempfile
import subprocess


from . import commonast
from ._spirv_constants import *


def parse_python(py_func):
    py_code = inspect.getsource(py_func)
    py_ast = commonast.parse(py_code)
    return py_ast


def generate_spirv(py_ast):
    g = SpirVGenerator(py_ast)
    g.generate()
    return g


STORAGE_CLASSES = dict(
    constant=StorageClass_UniformConstant,
    input=StorageClass_Input,
    uniform=StorageClass_Uniform,
    output=StorageClass_Output,
)


def str_to_words(s):
    b = s.encode()
    padding = 4 - (len(b) % 4)  # 4, 3, 2 or 1 -> always at least 1 for 0-termination
    b += padding * b"\x00"
    assert len(b) % 4 == 0 and b[-1] == 0, b
    # todo: endianness?
    words = []
    for i in range(0, len(b), 4):
        words.append(b[i : i + 4])
        #words.append(struct.unpack("<I", b[i : i + 4])[0])
    return words




class SpirVGenerator:
    """ Generate binary Spir-V code from a Python AST.
    """

    def __init__(self, py_ast):
        self._root = py_ast

        assert len(self._root.body_nodes) == 1 and isinstance(
            self._root.body_nodes[0], commonast.FunctionDef
        )

    def _init(self):

        # Section 2.4 of the Spir-V spec specifies the Logical Layout of a Module
        self._sections = {
            "capabilities": [],  # 1. All OpCapability instructions.
            "extensions": [],  # 2. Optional OpExtension instructions.
            "extension_imports": [],  # 3. Optional OpExtInstImport instructions.
            "memory_model": [],  # 4. The single required OpMemoryModel instruction.
            "entry_points": [],  # 5. All entry point declarations, using OpEntryPoint.
            "execution_modes": [],  # 6. All execution-mode declarations, using OpExecutionMode or OpExecutionModeId.
            "debug": [],  # 7. The debug instructions, which must be grouped in a specific following order.
            "annotations": [],  # 8. All annotation instructions, e.g. OpDecorate.
            "types": [],  # 9. All type declarations (OpTypeXXX instructions),
                          # all constant instructions, and all global
                          # variable declarations (all OpVariable instructions whose
                          # Storage Class is notFunction). This is the preferred
                          # location for OpUndef instructions, though they can also
                          # appear in function bodies. All operands in all these
                          # instructions must be declared before being used. Otherwise,
                          # they can be in any order. This section is the ﬁrst section
                          # to allow use of OpLine debug information.
            "function_defs": [],  # 10. All function declarations. A function
                                  # declaration is as follows.
                                  # a. Function declaration, using OpFunction.
                                  # b. Function parameter declarations, using OpFunctionParameter.
                                  # c. Function end, using OpFunctionEnd.
            "functions": [],  # 11. All function deﬁnitions (functions with a body).
                              # A function deﬁnition is as follows:
                              # a. Function deﬁnition, using OpFunction.
                              # b. Function parameter declarations, using OpFunctionParameter.
                              # c. Block, Block ...
                              # d. Function end, using OpFunctionEnd.
        }

        self._ids = {0: None}  # todo: I think this can simply be a counter ...
        self._type_info = {}  # type_name -> id
        self.scope_stack = []  # stack of dicts: name -> id, type, type_id
        # todo: can we do without a stack, pass everything into funcs?

    def generate(self):
        """ Generate the Spir-V code. After this, to_binary() can be used to
        produce the binary blob that represents the Spir-V module.
        """

        # Start clean
        self._init()

        # Define capabilities. Therea area lot more, and we probably should detect
        # the cases when we need to define them, and/or let the user define them somehow.
        self.gen_instruction("capabilities", OpCapability, Capability_Matrix)
        self.gen_instruction("capabilities", OpCapability, Capability_Shader)
        # self.gen_instruction("capabilities", OpCapability, Capability_Geometry)
        # self.gen_instruction("capabilities", OpCapability, Capability_Float16)
        # self.gen_instruction("capabilities", OpCapability, Capability_Float64)
        self.gen_instruction("capabilities", OpCapability, Capability_ImageBasic)

        # Define memory model (1 instruction)
        self.gen_instruction("memory_model", OpMemoryModel, AddressingModel_Logical, MemoryModel_Simple)

        # Define entry points
        self._main_id = self.create_id("main")
        self.gen_instruction(
            "entry_points", OpEntryPoint, ExecutionModel_Fragment, self._main_id, "main"
        )  # todo: arg1, arg2, pointers, that, are, used)

        # Define execution modes for each entry point
        self.gen_instruction(
            "execution_modes", OpExecutionMode, self._main_id, ExecutionMode_OriginLowerLeft
        )

        # Now walk the AST!
        self.scope_stack.append({})
        for node in self._root.body_nodes[0].body_nodes:
            self.parse(node)

    def to_text(self):
        """ Generate a textual (dis-assembly-like) representation.
        """
        lines = []

        def disp(pre, pro):
            pre = pre or ""
            line = str(pre.rjust(18)) + "  " + str(pro)
            lines.append(line)

        disp("header", "=" * 20)
        disp("MagicNumber", hex(MagicNumber))
        disp("Version", hex(Version))
        disp("VendorId", hex(0))
        disp("Bounds", len(self._ids))
        disp("Reserved", hex(0))

        for section_name, instructions in self._sections.items():
            disp(section_name, "=" * 20)
            for instruction in instructions:
                disp(None, instruction)

        return "\n".join(lines)

    def to_binary(self):
        """ Generated a bytes object representing the Spir-V module.
        """
        f = io.BytesIO()

        def write_word(w):
            if isinstance(w, bytes):
                assert len(w) == 4
                f.write(w)
            else:
                f.write(struct.pack("<I", w))

        # Write header
        write_word(MagicNumber)  # Magic number
        write_word(Version)  # SpirV version
        write_word(0)  # Vendor id - can be zero, let's use zero until we are registered
        write_word(len(self._ids))  # Bound (of ids)
        write_word(0)  # Reserved

        # Write instructions
        for instructions in self._sections.values():
            for opcode, *instr_words in instructions:
                words = []
                for word in instr_words:
                    if isinstance(word, str):
                        words.extend(str_to_words(word))
                    else:
                        words.append(word)
                write_word(((len(words) + 1) << 16) | opcode)
                for word in words:
                    write_word(word)

        return f.getvalue()

    def validate(self):
        """ Validate the generated code by running spirv-val from the Vulkan SDK.
        """
        filename = os.path.join(tempfile.gettempdir(), "x.spv")
        with open(filename, "wb") as f:
            f.write(self.to_binary())
        subprocess.check_call(["spirv-val", filename])

    ## Utils

    def gen_instruction(self, section_name, opcode, *words):
        self._sections[section_name].append((opcode, *words))

    def gen_func_instruction(self, opcode, *words):
        self._sections["functions"].append((opcode, *words))

    def create_id(self, type_id, xx=None):
        """ Get an id for a type, variable, function, etc.
        The name is optional, for debugging purposes.
        The "namespace" for id's is global to the whole shader/kernel.
        """
        # assert isinstance(type_id, int)
        id = len(self._ids)
        self._ids[id] = type_id
        return id

    def create_variable(self, type, name, storage_class):
        """ Create a variable in the current scope. Generates a variable
        definition instruction. Return the new id.
        """
        scope = self.scope_stack[-1]
        # Create type
        type_id = self.create_type_id(type)
        # Create pointer type thingy
        pointer_id = self.create_id(None)
        self.gen_instruction("types", OpTypePointer, pointer_id, storage_class, type_id)
        # Create the variable declaration
        id = self.create_id(type_id)
        self.gen_instruction("types", OpVariable, pointer_id, id, storage_class)
        scope[name] = name, id, type, type_id
        scope[id] = name, id, type, type_id
        return id

    def get_variable_info(self, name_or_id):
        """ Get (name, id, type, type_id) for the given variable name or id.
        """
        for scope in reversed(self.scope_stack):
            if name_or_id in scope:
                break
        else:
            if name_or_id == "gl_VertexIndex":
                return name_or_id, name_or_id, "int", self.create_type_id("int")
            else:
                raise NameError(f"Variable {name_or_id} not found.")
        return scope[name_or_id]  # name, id, type, type_id

    def create_type_id(self, type_name):
        """ Get the id for the given type_name. Generates a type
        definition instruction as needed.
        """
        if type_name in self._type_info:
            return self._type_info[type_name]
        if type_name == "void":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types", OpTypeVoid, type_id)
            return type_id
        elif type_name == "uint":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types", OpTypeInt, type_id, 32, 0)  # unsigned
            return type_id
        elif type_name == "int":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types", OpTypeInt, type_id, 32, 1)  # signed
            return type_id
        elif type_name == "float":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types", OpTypeFloat, type_id, 32)
            return type_id
        elif type_name.startswith("vec"):
            float_id = self.create_type_id("float")
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types",
                OpTypeVector, type_id, float_id, int(type_name[3:])
            )
            return type_id
        elif type_name.startswith("array"):
            _, n, sub_type_name = type_name.split("_", 2)
            # Handle subtype
            sub_type_id = self.create_type_id(sub_type_name)
            # Handle count
            # count_type_id = self.create_type_id("uint")
            # self.gen_instruction("types", OpConstant, count_type_id, 3)
            # Handle toplevel array type
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction("types", OpTypeArray, type_id, sub_type_id, n)
            return type_id

        else:
            raise NotImplementedError()

    def get_type_id(self, id):
        res = self._ids[id]
        # assert isinstance(res, int)
        # int for values, str for types
        return res


    ## The parse functions

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
        result_type_id = self.create_type_id("void")
        function_id = self._main_id  # self.create_id(result_type_id)

        # Generate the function type -> specify args and return value
        function_type_id = self.create_id(result_type_id)
        self.gen_instruction("types",
            OpTypeFunction, function_type_id, result_type_id, *[]
        )  # can haz arg ids

        # todo: also generate instructions in function_defs section

        # Generate function
        function_control = 0  # can specify whether it should inline, etc.
        self.gen_instruction(
            "functions", OpFunction, result_type_id, function_id, function_control, function_type_id
        )

        # A function must begin with a label
        self.gen_instruction("functions", OpLabel, self.create_id("label"))

        for sub in node.body_nodes:
            self.parse(sub)

        self.gen_instruction("functions", OpReturn)
        self.gen_instruction("functions", OpFunctionEnd)
        self.scope_stack.pop(-1)

    def parse_Assign(self, node):
        scope = self.scope_stack[-1]
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
            self.gen_func_instruction(OpStore, target_id, result_id, 0)

    def parse_Call(self, node):
        funcname = node.func_node.name
        arg_ids = [self.parse(arg) for arg in node.arg_nodes]
        if funcname in ("vec2", "vec3", "vec4"):
            composite_ids = []
            for id in arg_ids:
                type_id = self.get_type_id(id)
                type_name = self.get_type_id(type_id)  # todo: this is weird
                if type_name == "float":
                    composite_ids.append(id)
                elif type_name in ("vec2", "vec3", "vec4"):
                    for i in range(int(type_name[3:])):
                        type_id = self.create_type_id("float")
                        comp_id = self.create_id(type_id)
                        self.gen_func_instruction(
                            OpCompositeExtract, type_id, comp_id, id, i
                        )
                        composite_ids.append(comp_id)
                else:
                    raise TypeError(f"Cannot convert create vec4 from {type}")
            if len(composite_ids) != int(funcname[3:]):
                raise TypeError(
                    f"{funcname} did not expect {len(composite_ids)} elements"
                )
            type_id = self.create_type_id(funcname)
            result_id = self.create_id(type_id)
            self.gen_func_instruction(
                OpCompositeConstruct, type_id, result_id, *composite_ids
            )
            return result_id
        else:
            raise NotImplementedError()

    def parse_Name(self, node):
        name, id, type, type_id = self.get_variable_info(node.name)

        # todo: only load when the name is a pointer, and only once per func
        load_id = self.create_id(type_id)
        self.gen_func_instruction(OpLoad, type_id, load_id, id)
        return load_id

    def parse_Num(self, node):
        # todo: re-use constants
        if isinstance(node.value, int):
            type_id = self.create_type_id("int")
            result_id = self.create_id(type_id)
            self.gen_instruction(
                "types", OpConstant, type_id, result_id, struct.pack("<I", node.value)
            )
        elif isinstance(node.value, float):
            type_id = self.create_type_id("float")
            result_id = self.create_id(type_id)
            self.gen_instruction(
                "types", OpConstant, type_id, result_id, struct.pack("<f", node.value)
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

        type_id = self.create_type_id("vec2")
        result_id = self.create_id(type_id)

        return result_id
