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

def to_snake_case(camel_str):
    snake_str = camel_str[0].lower()
    for char in camel_str[1:]:
        if char.isupper():
            snake_str += "_"
        snake_str += char.lower()
    return snake_str


# this one liner actually does the trick
def keys_to_camel_case_converter(in_dict: dict, convert, cache=None) -> dict:
    skip_keys = ("constants",)
    return {
        to_camel_case(k): to_js(v, eager_converter=simple_js_accessor) # recursion on most members here
        if k not in skip_keys
        else to_js(v) # this is the member step for any keys we skip, so the default doesn't do any internal renaming 
        for k, v in in_dict.items()
    }


# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter

# TODO: do we need cache? or is that only for self referential objects?
# you have to do the recursion yourself...
def simple_js_accessor(value, convert, cache=None):
    if isinstance(value, classes.GPUObjectBase):
        return value._internal # type : JsProxy
    elif isinstance(value, structs.Struct):
        # for structs, we need to access the inner __dict__ object so pyodide can convert it to a JsProxy object
        # otherwise the dataclass with a ".type" prop will return the value of type(x) in js.
        return to_js(value.__dict__, eager_converter=simple_js_accessor, dict_converter=keys_to_camel_case_converter)
    elif isinstance(value, dict):
        # as we now are handed a dict, the inner values need to recursively also be converted
        # plus js expects camelCase for almost all keys here. there is exceptions in the function boy
        return to_js(keys_to_camel_case_converter(value, convert, cache), depth=1)
    elif isinstance(value, (list, tuple)):
        result = [to_js(v, eager_converter=simple_js_accessor) for v in value]
        return to_js(result, depth=1) # to make sure it's like an ArrayList?

    # this is the default?
    return convert(value) # or to_js(value)?

# TODO: can we implement our own variant of JsProxy and PyProxy, to_js and to_py? to work with pyodide and not around it?
# https://pyodide.org/en/stable/usage/type-conversions.html#type-translations
