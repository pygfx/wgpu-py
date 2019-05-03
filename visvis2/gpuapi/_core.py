
class GPUObject:

    _handle = None

    def __init__(self):
        self._handle = self._create_handle()

    def __del__(self):
        self._destroy()

    def _destroy(self):
        pass



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

