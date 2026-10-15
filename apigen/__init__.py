""" Code to generate the wgpu-py API.
"""

from download_idl import download_idl
from extract_idl import extract_idl
from format_json import format_json
from merge_json import merge_json
from write_api import write_api

actions = [download_idl, extract_idl, format_json, merge_json, write_api]
