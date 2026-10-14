import os
import tomllib

from apigen_utils import DEFINITIONS_DIR, read_file

FILES = [fname for fname in os.listdir(DEFINITIONS_DIR) if fname.endswith(".toml")]
BACKEND_FILES = ["diff_backend_pyodide.toml", "diff_backend_wgpu.toml"]


def check_all_definition_files():

    for fname in FILES:
        source = read_file("apigen", "definitions", fname)
        s = tomllib.loads(source)
