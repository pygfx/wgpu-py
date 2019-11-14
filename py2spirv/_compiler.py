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
import struct
import inspect
import tempfile
import subprocess

from . import _spirv_constants as cc


STORAGE_CLASSES = dict(
    constant=cc.StorageClass_UniformConstant,
    input=cc.StorageClass_Input,
    uniform=cc.StorageClass_Uniform,
    output=cc.StorageClass_Output,
)


def str_to_words(s):
    b = s.encode()
    padding = 4 - (len(b) % 4)  # 4, 3, 2 or 1 -> always at least 1 for 0-termination
    b += padding * b"\x00"
    assert len(b) % 4 == 0 and b[-1] == 0, b
    words = []
    for i in range(0, len(b), 4):
        words.append(b[i : i + 4])
        #words.append(struct.unpack("<I", b[i : i + 4])[0])
    return words


class IdInt(int):

    def __repr__(self):
        return "%" + super().__repr__()


class BaseSpirVCompiler:
    """ Generate binary Spir-V code from a Python AST.
    """

    def __init__(self, py_func):
        if not inspect.isfunction(py_func):
            raise TypeError("Python to SpirV Compiler needs a Python function.")
        self._py_func = py_func
        self._prepare()

    def _prepare(self):
        # Subclass should e.g. do some parsing here. Don't do too much work yet.
        raise NotImplementedError()

    def _generate(self):
        # Subclasses should generate SpirV here
        raise NotImplementedError()

    def _init(self):

        self._ids = {0: None}  # todo: I think this can simply be a counter ...
        self._type_name_to_id = {}
        self._type_id_to_name = {}
        self.scope_stack = []  # stack of dicts: name -> id, type, type_id
        # todo: can we do without a stack, pass everything into funcs?

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

    def generate(self):
        """ Generate the Spir-V code. After this, to_binary() can be used to
        produce the binary blob that represents the Spir-V module.
        """

        # Start clean
        self._init()

        # Define capabilities. Therea area lot more, and we probably should detect
        # the cases when we need to define them, and/or let the user define them somehow.
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Matrix)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Shader)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Geometry)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float16)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float64)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_ImageBasic)

        # Define memory model (1 instruction)
        self.gen_instruction("memory_model", cc.OpMemoryModel, cc.AddressingModel_Logical, cc.MemoryModel_Simple)

        # Define entry points
        self._main_id = self.create_id("main")
        self.gen_instruction(
            "entry_points", cc.OpEntryPoint, cc.ExecutionModel_Fragment, self._main_id, "main"
        )  # todo: arg1, arg2, pointers, that, are, used)

        # Define execution modes for each entry point
        self.gen_instruction(
            "execution_modes", cc.OpExecutionMode, self._main_id, cc.ExecutionMode_OriginLowerLeft
        )

        self._generate()

    def to_text(self):
        """ Generate a textual (dis-assembly-like) representation.
        """
        lines = []
        edge = 22

        def disp(pre, pro):
            pre = pre or ""
            line = str(pre.rjust(edge)) + str(pro)
            lines.append(line)

        disp("header ".ljust(edge, "-"), "")
        disp("MagicNumber: ", hex(cc.MagicNumber))
        disp("Version: ", hex(cc.Version))
        disp("VendorId: ", hex(0))
        disp("Bounds: ", len(self._ids))
        disp("Reserved: ", hex(0))

        types = set()
        for section_name, instructions in self._sections.items():
            #disp(section_name.upper(), "-" * 20)
            disp((section_name + " ").ljust(edge, "-"), "")
            for instruction in instructions:
                instruction_str = repr(instruction[0])
                ret = None
                for i in instruction[1:]:
                    if isinstance(i, IdInt):
                        i_str = "%" + self._type_id_to_name.get(i, str(i))
                        if i_str not in types:
                            types.add(i_str)
                            ret = i_str + " = "
                            i_str = "(" + repr(i) + ")"
                        instruction_str += " " + i_str
                    else:
                        instruction_str += " " + repr(i)
                disp(ret, instruction_str)

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
        write_word(cc.MagicNumber)  # Magic number
        write_word(cc.Version)  # SpirV version
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

    def disassble(self):
        """ Disassemble the generated binary code using spirv-dis, and return as a string.
        This produces a result similar to to_text(), but to_text() is probably more
        informative.

        Needs Spir-V tools, which can easily be obtained by installing the Vulkan SDK.
        """
        filename = os.path.join(tempfile.gettempdir(), "x.spv")
        with open(filename, "wb") as f:
            f.write(self.to_binary())
        try:
            stdout = subprocess.check_output(["spirv-dis", filename], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            e = "Could not disassemle Spir-V:\n" + err.output.decode()
            raise Exception(e)
        else:
            return stdout.decode()

    def validate(self):
        """ Validate the generated binary code by running spirv-val
        .
        Needs Spir-V tools, which can easily be obtained by installing the Vulkan SDK.
        """
        filename = os.path.join(tempfile.gettempdir(), "x.spv")
        with open(filename, "wb") as f:
            f.write(self.to_binary())
        try:
            stdout = subprocess.check_output(["spirv-val", filename], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            e = "Spir-V invalid:\n" + err.output.decode()
            raise Exception(e)
        else:
            print("Spir-V seems valid:\n" + stdout.decode())

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
        id = IdInt(len(self._ids))
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
        self.gen_instruction("types", cc.OpTypePointer, pointer_id, storage_class, type_id)
        # Create the variable declaration
        id = self.create_id(type_id)
        self.gen_instruction("types", cc.OpVariable, pointer_id, id, storage_class)
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
        # Already know this type?
        if type_name in self._type_name_to_id:
            return self._type_name_to_id[type_name]

        if type_name == "void":
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types", cc.OpTypeVoid, type_id)
        elif type_name == "uint":
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types", cc.OpTypeInt, type_id, 32, 0)  # unsigned
        elif type_name == "int":
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types", cc.OpTypeInt, type_id, 32, 1)  # signed
        elif type_name == "float":
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types", cc.OpTypeFloat, type_id, 32)
        elif type_name.startswith("vec"):
            float_id = self.create_type_id("float")
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types",
                cc.OpTypeVector, type_id, float_id, int(type_name[3:])
            )
        elif type_name.startswith("array"):
            _, n, sub_type_name = type_name.split("_", 2)
            # Handle subtype
            sub_type_id = self.create_type_id(sub_type_name)
            # Handle count
            # count_type_id = self.create_type_id("uint")
            # self.gen_instruction("types", OpConstant, count_type_id, 3)
            # Handle toplevel array type
            type_id = self.create_id(type_name, None)
            self.gen_instruction("types", cc.OpTypeArray, type_id, sub_type_id, n)
        else:
            raise NotImplementedError()

        self._type_id_to_name[type_id] = type_name
        self._type_name_to_id[type_name] = type_id
        return type_id

    def get_type_id(self, id):
        res = self._ids[id]
        # assert isinstance(res, int)
        # int for values, str for types
        return res

