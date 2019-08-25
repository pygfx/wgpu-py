"""
A compiler to translate Python code to SpirV.

References:
* https://www.khronos.org/registry/spir-v/

"""

import os
import io
import ast
import struct
import inspect

import vulkan as vk
from . import commonast

from ._spirv_constants import *


def vertex_shader():

    return 3 + 4


vec3 = "vec3"
vec4 = "vec4"

STORAGE_CLASSES = dict(
    constant=StorageClass_UniformConstant,
    input=StorageClass_Input,
    uniform=StorageClass_Uniform,
    output=StorageClass_Output,
)


def fragment_shader():
    # version 450
    # extension GL_ARB_separate_shader_objects : enable

    fragColor = input(vec3)
    outColor = output(vec4)

    # layout(location = 0) in vec3 fragColor;
    # layout(location = 0) out vec4 outColor;

    def main():
        outColor = vec4(fragColor, 0.5)


def get_shader_from_spirv(device, code):
    createInfo = vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(code),
        pCode=code,
    )
    return vk.vkCreateShaderModule(device, createInfo, None)


def get_vert_shader(device):
    nogit = os.path.abspath(os.path.join(__file__, "..", "..", "..", "nogit"))
    with open(os.path.join(nogit, "hello_triangle_vert.spv"), "rb") as sf:
        code = sf.read()
    return get_shader_from_spirv(device, code)


def get_frag_shader(device):
    # nogit = os.path.abspath(os.path.join(__file__, "..", "..", "..", "nogit"))
    # with open(os.path.join(nogit, "hello_triangle_frag.spv"), "rb") as sf:
        # code = sf.read()

    x = Python2SpirVCompiler(fragment_shader)
    x.start_parsing()
    code = x.dump()

    return get_shader_from_spirv(device, code)


##


def str_to_words(s):
    b = s.encode() + b"\x00"
    b += ((4 - (len(b) % 4)) % 4) * b"\x00"
    assert len(b) % 4 == 0
    # todo: endianness?
    words = []
    for i in range(0, len(b), 4):
        words.append(struct.unpack("<I", b[i : i + 4])[0])
    return words


class Python2SpirVCompiler:
    def __init__(self, func):
        pycode = inspect.getsource(func)
        self.root = commonast.parse(pycode)

        self._ids = {0: None}  # todo: I think this can simply be a counter ...
        self._type_info = {}  # type_name -> id
        self.scope_stack = []  # stack of dicts: name -> id, type, type_id
        # todo: can we do without a stack, pass everything into funcs?

        self.instructions_pre = []
        self.instructions = []

        assert len(self.root.body_nodes) == 1 and isinstance(
            self.root.body_nodes[0], commonast.FunctionDef
        )

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
        self.gen_instruction_pre(OpTypePointer, pointer_id, storage_class, type_id)
        # Create the variable declaration
        id = self.create_id(type_id)
        self.gen_instruction(OpVariable, pointer_id, id, storage_class)
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
            self.gen_instruction_pre(OpTypeVoid, type_id)
            return type_id
        elif type_name == "int":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction_pre(OpTypeInt, type_id, 32)
            return type_id
        elif type_name == "float":
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction_pre(OpTypeFloat, type_id, 32)
            return type_id
        elif type_name.startswith("vec"):
            float_id = self.create_type_id("float")
            self._type_info[type_name] = type_id = self.create_id(type_name, None)
            self.gen_instruction_pre(
                OpTypeVector, type_id, float_id, int(type_name[3:])
            )
            return type_id
        else:
            raise NotImplementedError()

    def get_type_id(self, id):
        res = self._ids[id]
        # assert isinstance(res, int)
        # int for values, str for types
        return res

    def gen_instruction_pre(self, opcode, *words):
        self.instructions_pre.append((opcode, *words))

    def gen_instruction(self, opcode, *words):
        self.instructions.append((opcode, *words))

    def dump(self):
        f = io.BytesIO()

        def write_word(w):
            if isinstance(w, bytes):
                assert len(w) == 4
                f.write(w)
            else:
                f.write(struct.pack("<I", w))

        # Write header
        write_word(0x07230203)  # Magic number
        write_word(0x00010000)  # SpirV version
        write_word(0x02820282)  # hex(sum([ord(x) for x in "Python"]))
        write_word(len(self._ids))  # Bound (of ids)
        write_word(0)  # Reserved

        # Write instructions
        instructions = self.instructions_pre + self.instructions
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

    def start_parsing(self):
        self.gen_instruction_pre(OpCapability, 1)
        # self.gen_instruction_pre(OpExtension
        # self.gen_instruction_pre(OpExtInstImport
        self.gen_instruction_pre(OpMemoryModel, 0, 0)  # todo: 0 or 1 for memory model?

        # Create entry point
        self._main_id = self.create_id("main")
        self.gen_instruction_pre(
            OpEntryPoint, ExecutionModel_Fragment, self._main_id, "main"
        )  # todo: arg1, arg2, pointers, that, are, used)

        if True:  # fragment shader
            self.gen_instruction_pre(
                OpExecutionMode, self._main_id, ExecutionMode_OriginLowerLeft
            )

        self.scope_stack.append({})
        for node in self.root.body_nodes[0].body_nodes:
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
        result_type_id = self.create_type_id("void")
        function_id = self._main_id  # self.create_id(result_type_id)

        # Generate the function type -> specofy args and return value
        function_type_id = self.create_id(result_type_id)
        self.gen_instruction_pre(
            OpTypeFunction, function_type_id, result_type_id, *[]
        )  # can haz arg ids

        # Generate function
        function_control = 0  # can specify whether it should inline, etc.
        self.gen_instruction(
            OpFunction, result_type_id, function_id, function_control, function_type_id
        )

        # A function must begin with a label
        self.gen_instruction(OpLabel, self.create_id("label"))

        for sub in node.body_nodes:
            self.parse(sub)

        self.gen_instruction(OpReturn)
        self.gen_instruction(OpFunctionEnd)
        self.scope_stack.pop(-1)

    def parse_Assign(self, node):
        scope = self.scope_stack[-1]
        assert len(node.target_nodes) == 1
        varname = node.target_nodes[0].name

        if len(self.scope_stack) == 1:

            if isinstance(node.value_node, commonast.Call):
                funcname = node.value_node.func_node.name
                argnames = [arg.name for arg in node.value_node.arg_nodes]
                if funcname in ("input", "output"):
                    assert len(argnames) == 1
                    vartype = argnames[0]
                    self.create_variable(vartype, varname, STORAGE_CLASSES[funcname])
                else:
                    # scope[varname] = ??
                    raise NotImplementedError()

        else:
            root = self.scope_stack[0]
            assert varname in root, "can only use global vars yet"
            _, target_id, _, _ = self.get_variable_info(varname)

            result_id = self.parse(node.value_node)
            assert result_id
            self.gen_instruction(OpStore, target_id, result_id, 0)

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
                        self.gen_instruction(
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
            self.gen_instruction(
                OpCompositeConstruct, type_id, result_id, *composite_ids
            )
            return result_id
        else:
            raise NotImplementedError()

    def parse_Name(self, node):
        name, id, type, type_id = self.get_variable_info(node.name)

        # todo: only load when the name is a pointer, and only once per func
        load_id = self.create_id(type_id)
        self.gen_instruction(OpLoad, type_id, load_id, id)
        return load_id

    def parse_Num(self, node):
        # todo: re-use constants
        if isinstance(node.value, int):
            type_id = self.create_type_id("int")
            result_id = self.create_id(type_id)
            self.gen_instruction_pre(
                OpConstant, type_id, result_id, struct.pack("<I", node.value)
            )
        elif isinstance(node.value, float):
            type_id = self.create_type_id("float")
            result_id = self.create_id(type_id)
            self.gen_instruction_pre(
                OpConstant, type_id, result_id, struct.pack("<f", node.value)
            )
        return result_id


if __name__ == "__main__":
    x = Python2SpirVCompiler(fragment_shader)
    x.start_parsing()
    with open("b.spv", "wb") as f:
        f.write(x.dump())
