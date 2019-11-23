"""
A compiler to translate Python code to SpirV. A compiler can be
considered to exists of two parts: a parser to go from one code (Python)
to an AST, and a generator to transform that AST into another code
(SpirV). In this case the parsing is done using Python's builtin ast
module, so the main part of this module comprises the SpirV generation.

References:
* https://www.khronos.org/registry/spir-v/


Type info:
* Basic types are boolean, int, float. Latter two are numerics, all three are scalars.
* Vector is two or more values of scalars (float, int, bool). For lengt > 4 need capabilities.
* Matrix is 2, 3, or 4 float vectors (each vector is a column).
* Array is homogeneous collection of non-void-type objects.
* Structure is heterogeneous collection of non-void-type objects.
* imagee, sampler, ...

"""

import os
import io
import struct
import inspect
import tempfile
import subprocess

from . import _spirv_constants as cc
from . import _types

# todo: add debug info, most notably line numbers, but can also do source code and object names!




# %%

# Storage class members, used in OpTypePointer, OpTypeForwardPointer, Opvariable, OpGenericCastToPtrExplicit
STORAGE_CLASSES = dict(
    uniform_constant=cc.StorageClass_UniformConstant,
    input=cc.StorageClass_Input,
    uniform=cc.StorageClass_Uniform,
    output=cc.StorageClass_Output,
    private=cc.StorageClass_Private,  # a constant within this module
    function=cc.StorageClass_Function,  # scoped to the current function invokation
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

        self._ids = {0: None}  # maps id -> info. For objects, info is a type in _types
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

    def generate(self, execution_model="vertex"):
        """ Generate the Spir-V code. After this, to_binary() can be used to
        produce the binary blob that represents the Spir-V module.
        """

        # todo: somehow derive execution_model from the function itself
        if execution_model.lower() == "vertex":
            execution_model = cc.ExecutionModel_Vertex
        elif execution_model.lower() == "fragment":
            execution_model = cc.ExecutionModel_Fragment
        else:
            raise ValueError(f"Unknown execution model: {execution_model}")

        # Start clean
        self._init()

        # Define capabilities. Therea area lot more, and we probably should detect
        # the cases when we need to define them, and/or let the user define them somehow.
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Matrix)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Shader)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Geometry)
        # self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float16)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_Float64)
        self.gen_instruction("capabilities", cc.OpCapability, cc.Capability_ImageBasic)

        # Define memory model (1 instruction)
        self.gen_instruction("memory_model", cc.OpMemoryModel, cc.AddressingModel_Logical, cc.MemoryModel_Simple)

        # Define entry points
        # Note that we must add the ids of all used OpVariables that this entrypoint uses.
        self._entry_point_id = self.create_id("main")
        self.gen_instruction(
            "entry_points", cc.OpEntryPoint, execution_model, self._entry_point_id, "main"
        )

        # Define execution modes for each entry point
        #self.gen_instruction(
        #    "execution_modes", cc.OpExecutionMode, self._entry_point_id, cc.ExecutionMode_OriginLowerLeft
        #)

        self._generate()

        # Move OpVariable to the start of a function
        func_instructions = self._sections["functions"]
        insert_point = -1
        for i in range(len(func_instructions)):
            if func_instructions[i][0] == cc.OpFunction:
                insert_point = -1
            elif insert_point < 0:
                if func_instructions[i][0] == cc.OpLabel:
                    insert_point = i + 1
            elif func_instructions[i][0] == cc.OpVariable:
                func_instructions.insert(insert_point, func_instructions.pop(i))
                insert_point += 1

        # Get ids of global variables
        global_OpVariable_s = []
        for instr in self._sections["types"]:
            if instr[0] == cc.OpVariable:
                global_OpVariable_s.append(instr[2])
        # We assume one function, so all are used in our single function
        self._sections["entry_points"][0] = self._sections["entry_points"][0] + tuple(global_OpVariable_s)

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
                        if instruction[0] == cc.OpDecorate:
                            pass
                        elif i_str not in types:
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
            e = "Could not disassemble Spir-V:\n" + err.output.decode()
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

    def create_id(self, description):
        """ Get a new id for a type, variable, object, pointer, function, etc.
        The "namespace" for id's is global to the whole shader/kernel.
        """
        # Also allow a name to generate OpName
        id = IdInt(len(self._ids))
        self._ids[id] = description
        return id

    def create_object(self, the_type):
        """ Create id for a new object. Returns (id, type_id).
        """
        assert isinstance(the_type, type), f"create_id requires a type, not {the_type}"
        assert issubclass(the_type, _types.SpirVType), f"not a spirv type: {the_type}"
        type_id = self.get_type_id(the_type)
        id = self.create_id(the_type)
        return id, type_id

    def get_type_from_id(self, id):
        """ Get the type of a given object id.
        """
        t = self._ids[id]
        assert isinstance(t, type)
        return t

    def create_variable(self, type, name, storage_class):
        """ Create a variable in the current scope. Generates a variable
        definition instruction. Return the new id.
        """
        scope = self.scope_stack[-1]
        # Create type
        type_id = self.get_type_id(type)
        # Create pointer type thingy
        pointer_id = self.create_id("pointer")
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
                return name_or_id, name_or_id, "int", self.get_type_id("int")
            else:
                raise NameError(f"Variable {name_or_id} not found.")
        return scope[name_or_id]  # name, id, type, type_id

    def get_type_id(self, the_type):
        """ Get the id for the given type_name. Generates a type
        definition instruction as needed.
        """
        assert isinstance(the_type, type), f"create_id requires a type, not {the_type}"
        assert issubclass(the_type, _types.SpirVType), f"not a spirv type: {the_type}"
        # todo: check if is concrete type

        # Already know this type?
        if the_type.__name__ in self._type_name_to_id:
            return self._type_name_to_id[the_type.__name__]

        if issubclass(the_type, _types.void):
            type_id = self.create_id(the_type)
            self.gen_instruction("types", cc.OpTypeVoid, type_id)
        elif issubclass(the_type, _types.boolean):
            type_id = self.create_id(the_type)
            self.gen_instruction("types", cc.OpTypeBool, type_id)
        elif issubclass(the_type, _types.Int):
            type_id = self.create_id(the_type)
            bits = 32
            if issubclass(the_type, _types.i16):
                # todo: need OpCapability
                bits = 16
            elif issubclass(the_type, _types.i64):
                bits = 64
            self.gen_instruction("types", cc.OpTypeInt, type_id, bits, 0)  # no signedness semantics
        elif issubclass(the_type, _types.Float):
            type_id = self.create_id(the_type)
            bits = 32
            if issubclass(the_type, _types.f16):
                # todo: need OpCapability
                bits = 16
            elif issubclass(the_type, _types.f64):
                bits = 64
            self.gen_instruction("types", cc.OpTypeFloat, type_id, bits)
        elif issubclass(the_type, _types.Vector):
            sub_type_id = self.get_type_id(the_type.subtype)
            type_id = self.create_id(the_type)
            self.gen_instruction("types",
                cc.OpTypeVector, type_id, sub_type_id, the_type.length)
        elif issubclass(the_type, _types.Matrix):
            raise NotImplementedError()
            # OpTypeMatrix
        elif issubclass(the_type, _types.Array):
            count = the_type.length
            sub_type_id = self.get_type_id(the_type.subtype)
            # Handle count
            count_type_id = self.get_type_id(_types.i32)
            count_id = self.create_id("array_count")
            self.gen_instruction("types", cc.OpConstant, count_type_id, count_id, count)
            # Handle toplevel array type
            type_id = self.create_id(the_type)
            self.gen_instruction("types", cc.OpTypeArray, type_id, sub_type_id, count_id)
            # Also see OpTypeRuntimeArray when length is not known at compile time (use OpArrayLength)
        elif issubclass(the_type, _types.Struct):
            raise NotImplementedError()
            # OpTypeStruct
        else:
            raise NotImplementedError(the_type)

        # todo: also OpTypeImage and OpTypeSampledImage
        self._type_id_to_name[type_id] = the_type.__name__
        self._type_name_to_id[the_type.__name__] = type_id
        return type_id
