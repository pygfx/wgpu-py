"""
Helper functions for dealing with pyodide for the js webgpu backend.
"""

from ... import classes, structs
from pyodide.ffi import to_js, JsProxy, JsArray, jsnull
import js
from typing import Callable, get_type_hints


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

# TODO: clean this up before readying for merge!
# 2. check if the js to python roundtrip is needed
# 4. either try the built in cache argument, or use @cache or something... memory is slower than math tho - so maybe if we have a benchmark to compare it to.

def _convert_struct(struct_to_convert:structs.Struct, convert:Callable, cache:dict|None) -> JsProxy:
    """
    convert a `wgpu.structs.Struct` into it's js equivalent descriptor
    """
    result = {}
    for struct_key, struct_member in struct_to_convert.items():
        camel_key = to_camel_case(struct_key)
        if type(struct_member) is dict and "limits" in struct_key.lower():
            # if it's just a dict like limits, we still need to convert the keys to camelCase.
            # limits aren't literals but record<DOMString>, and we don't enumerate them in any python files.
            struct_member = {to_camel_case(limit_key): limit_value for limit_key, limit_value in struct_member.items()}
            # however stuff like constants keys are USVString which don't need the camel case conversion... I don't think we keep track of that either.

        # if there is a dict further down... we still need to fix those keys (if only strcut creation were recursive...)
        elif isinstance(struct_member, dict):
            _struct_types = get_type_hints(type(struct_to_convert)) # not always needed, so we could move this up or lazy load it.
            if len(struct_member) == 0:
                # I think because of the round trip below, None and {} don't get converted as expected... js.Object.new() seems to also work.
                result[camel_key] = js.Object.new() # jsnull is problematic for empty constants?
                continue # actually skip the back translation?

            # this should hopefully work on the first attempt, there is a 2nd attempt with dict that might work as a fallback
            # if object is included we should avoid that too.
            # this is slow on the BufferBinding entry as this has to fail up to 6 time before going through. I want to compare the numbers.
            member_types = _struct_types.get(struct_key).__args__
            for candidate_type in member_types:
                # alternatively, we could load the type hints of the candidate and check if keys are the same?
                # but troublsome with optionals. unless all() might do it.
                try:
                    member_as_struct = candidate_type(**struct_member)
                    # print(f"struct cast successful {struct_member} --> {member_as_struct}")
                    break # on success we should exit the candidate loop and continue with other struct members
                except TypeError as te: # type error is too generic I feel, I want "unexpected keyword argument"
                    # print(te, f" in {struct_to_convert.__class__.__name__}.{struct_key} tried to convert {struct_member=} into {candidate_type}")
                    # bascially try again (for GPUBindingResource mostly)
                    continue
            else:
                # this this freakishly not get reached because the break is in a try/except?
                print(f"failed to convert {struct_member=} into any of the candidate types {member_types} for {struct_to_convert.__class__.__name__}.{struct_key}")

            # we do recursive call (and round trip) with the original varible name at the bottom, so let's hand this down
            struct_member = member_as_struct

        # if there is a list of dicts... it will still call the the default sequence converter and then dict converter...
        elif isinstance(struct_member, (list)): #maybe tuple too?
            if struct_member and isinstance(struct_member[0], dict): # assume all elements are the same type
                member_type = get_type_hints(type(struct_to_convert)).get(struct_key).__args__[0].__args__[0] # -> sequence of union struct|dict
                struct_member = [member_type(**item) for item in struct_member]
                # the recursion is handled by the caller...
            else:
                # could be a list of other objects like GPUBindGroupLayout for example also RenderPassColorAttachment
                pass

        # print("initial call to down_convert", struct_member)
        down_convert = to_js(struct_member, eager_converter=simple_js_accessor) # recursive call for all trivial members, "base case"?
        # print("first result of down_convert", down_convert)
        # TODO: can we avoid this round trip because: failed to read 'type' property from GPUBufferBindingLayout: value 'dict' is not a valid enum -> it reads the .type property and not the field...
        # I tried different dict_converter... but I think they don't matter for the eager converter...
        # print(f"round trip needed for {struct_member} -> {down_convert}")
        # down_convert = to_js(down_convert.to_py(depth=1), depth=1) if hasattr(down_convert, "to_py") else down_convert
        # print("final result of down_convert:", down_convert)
        result[camel_key] = down_convert
    # print(f"struct conversion result: {struct_to_convert} -->>> {result}")
    return to_js(result) # to make it an Object and not a JsProxy

# this one liner almost works... but we can't have undefined values I think... so maybe we can translate those into something else
def _dict_keys_to_camel_case(in_dict) -> dict:
    skip_keys = ("constants", )
    return {to_camel_case(k): to_js(v, eager_converter=simple_js_accessor) if k not in skip_keys else to_js(v) for k, v in in_dict.items()}

# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter
# you have to do the recursion yourself...
def simple_js_accessor(value, convert, cache=None):
    # print("simple_js_accessor", value, type(value), dir(value))
    if isinstance(value, classes.GPUObjectBase):
        # print("GPUObjectBase detected", value)
        return value._internal # type : JsProxy
    elif isinstance(value, structs.Struct):
        return to_js(value.__dict__, eager_converter=simple_js_accessor)
    elif isinstance(value, dict):
        return to_js(_dict_keys_to_camel_case(value), depth=1)
    elif isinstance(value, (list, tuple)):
        # print("list detected", value, len(value))
        result = [to_js(v, eager_converter=simple_js_accessor) for v in value]
        # print("list conversion result", result)
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
