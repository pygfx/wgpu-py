import os
import tempfile
import subprocess


class SpirVModule:
    """ Representation of a SpirV module. Is basically a wrapper around the
    source input and the bytes representing the actual SpirV code.
    """

    def __init__(self, input, binary, description):
        self._input = input
        self._binary = binary
        self._description = description

    def __repr__(self):
        return f"<SpirVModule {self._description} at 0x{hex(id(self))}>"

    @property
    def input(self):
        """ The input used to produce this SpirV module.
        """
        return self._input

    def to_bytes(self):
        """ Return the binary representation of the SpirV module.
        """
        return self._binary

    def disassble(self):
        """ Disassemble the generated binary code using spirv-dis, and return as a string.
        This produces a result similar to to_text(), but to_text() is probably more
        informative.

        Needs Spir-V tools, which can easily be obtained by installing the Vulkan SDK.
        But you probably don't want to call this in end-user code.
        """
        filename = os.path.join(tempfile.gettempdir(), "x.spv")
        with open(filename, "wb") as f:
            f.write(self.to_bytes())
        try:
            stdout = subprocess.check_output(
                ["spirv-dis", filename], stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as err:
            e = "Could not disassemble Spir-V:\n" + err.output.decode()
            raise Exception(e)
        else:
            return stdout.decode()

    def validate(self):
        """ Validate the generated binary code by running spirv-val.

        Needs Spir-V tools, which can easily be obtained by installing the Vulkan SDK.
        But you probably don't want to call this in end-user code.
        """
        filename = os.path.join(tempfile.gettempdir(), "x.spv")
        with open(filename, "wb") as f:
            f.write(self.to_bytes())
        try:
            stdout = subprocess.check_output(
                ["spirv-val", filename], stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as err:
            out = err.output.decode()
        else:
            out = stdout.decode().strip()
        if out:
            raise Exception(f"Spir-V {self._description} invalid:\n{out}")
        else:
            print(f"Spir-V {self._description} seems valid!")
