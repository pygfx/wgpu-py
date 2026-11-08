"""Code to generate the wgpu-py API."""

from download_idl import download_idl
from extract_idl import extract_idl
from format_defs import format_defs
from merge_defs import merge_defs
from patch_api import patch_api
from write_api import write_api
from write_js_api import write_js_api

actions = [
    download_idl,
    extract_idl,
    format_defs,
    merge_defs,
    patch_api,
    write_api,
    write_js_api,
]
