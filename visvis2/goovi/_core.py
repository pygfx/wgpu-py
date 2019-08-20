class GPUObject:
    """ The base class for all GPU objects.
    """

    _handle = None

    def __init__(self):
        self._handle = 0
        raise NotImplementedError()

    def __del__(self):
        try:
            self._destroy()
        except Exception as err:
            print(f"Error destroying {self.__class__.__name__} object: {str(err)}")

    def _destroy(self):
        pass


# todo: perhaps the python vulkan lib has someting like this?
def struct_to_dict(s):
    d = {}
    if repr(s).startswith("<vulkan") and hasattr(s, "obj"):
        for key in dir(s.obj):
            val = getattr(s, key)
            if repr(val).startswith("<cdata"):
                val = struct_to_dict(val)
            d[key] = val
    elif repr(s).startswith("<cdata"):
        for key in dir(s):
            val = getattr(s, key)
            if repr(val).startswith("<cdata"):
                val = struct_to_dict(val)
            d[key] = val
    return d
