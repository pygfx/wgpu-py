"""
Helper functions for dealing with pyodide for the js webgpu backend.
"""

from ... import classes, structs
from pyodide.ffi import to_js


def to_camel_case(snake_str):
    components = snake_str.split('_')
    res = components[0] + ''.join(x.title() for x in components[1:])
    # maybe keywords are a problem?
    # https://pyodide.org/en/stable/usage/faq.html#how-can-i-access-javascript-objects-attributes-in-python-if-their-names-are-python-keywords
    # if res in ["type", "format"]:
    #     res += "_"
    return res


# TODO: clean this up before reading for merge!

# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter
# you have to do the recursion yourself...
def simple_js_accessor(value, convert, cache=None):
    # print("simple_js_accessor", value, type(value), dir(value))
    if isinstance(value, classes.GPUObjectBase):
        # print("GPUObjectBase detected", value)
        return value._internal # type : JsProxy
    elif isinstance(value, structs.Struct):
        result = {}
        for k, v in value.items():
            camel_key = to_camel_case(k)
            # if there is a dict further down... we still need to fix those keys
            if isinstance(v, dict):
                if len(v) == 0:
                    result[camel_key] = to_js(v, depth=1) # empty dicts need special treatment?
                    continue
                if(k == "resource"): # this one is a more complex type.... https://www.w3.org/TR/webgpu/#typedefdef-gpubindingresource
                    # print("struct with resource dict detected", k, v)
                    v = structs.BufferBinding(**v)
                    # print("RESOURCE AS A STRUCT:", v)
                    down_convert = to_js(v, eager_converter=simple_js_accessor)
                    down_convert = to_js(down_convert.to_py(depth=1), depth=1) if hasattr(down_convert, "to_py") else down_convert
                    result[camel_key] = down_convert
                    # print("called convert(v) on RESOURCE STRUCT", result[camel_key])
                    continue
                # print("struct with dict detected", value, k, v)
                # print(dir(value))
                v_struct_type_name = value.__annotations__[k].partition("Struct")[0] # will not work if there is more than two options -.-
                # print("likely v struct type_name", v_struct_type_name)
                v_struct_type = structs.__dict__[v_struct_type_name] # because the annotation is just a string... doesn't feel great
                # print("likely v struct type", v_struct_type)
                v = v_struct_type(**v)
                # print("converted to struct", v)

            # if there is a list of dicts... it will still call the the default sequence converter and then dict converter...
            elif isinstance(v, (list)): #maybe tuple too?
                if v and isinstance(v[0], dict): # assume all elements are the same type too and non empty?
                    # print("struct with list detected", value, k, v)
                    v_struct_type_name = value.__annotations__[k].removeprefix("Sequence[").partition("Struct")[0]
                    # print("likely v struct type_name", v_struct_type_name)
                    v_struct_type = structs.__dict__[v_struct_type_name]
                    # print("likely v struct type", v_struct_type)
                    v = [v_struct_type(**item) for item in v]
                    # print("converted to list of struct", v)
                else:
                    # could be a list of other objects like GPUBindGroupLayout for example.
                    pass
            # print("initial call to down_convert", v)
            down_convert = to_js(v, eager_converter=simple_js_accessor)
            # print("first result of down_convert", down_convert, dir(down_convert))
            down_convert = to_js(down_convert.to_py(depth=1), depth=1) if hasattr(down_convert, "to_py") else down_convert
            # print("final result of down_convert", down_convert)
            result[camel_key] = down_convert
        # print("struct conversion result: ", type(result), result)
        return result

    elif isinstance(value, (list, tuple)):
        result = [to_js(v, eager_converter=simple_js_accessor) for v in value]
        return to_js(result, depth=1) # to make sure it's like an ArrayList?
    # this might recursively call itself...
    # maybe use a map? or do a dict_converted?
    # elif isinstance(value, dict):
    #     result = {}
    #     # cache(value, result)
    #     for k, v in value.items():
    #         camel_key = to_camel_case(k) if isinstance(k, str) else k
    #         result[camel_key] = convert(v)
    #     if len(result) == 0:
    #         return Object.new() # maybe this?
        # let's hope this is only ever reached when all the contents are already converted.
        # map = Map.new(result.items())
        # return Object.fromEntries(map)
    # print("simple_js_accessor default", value, type(value))
    return convert(value) # or to_js(value)?

# TODO: can we implement our own variant of JsProxy and PyProxy, to_js and to_py? to work with pyodide and not around it?
# https://pyodide.org/en/stable/usage/type-conversions.html#type-translations
