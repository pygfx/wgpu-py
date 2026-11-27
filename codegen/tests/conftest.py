"""Make sure the codegen dir and the repo root are on the path, so that
the tests can import the codegen modules (and the wgpu package) directly.
"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", ".."))
codegen_dir = os.path.abspath(os.path.join(here, ".."))
for path in (repo_root, codegen_dir):
    if path not in sys.path:
        sys.path.insert(0, path)
