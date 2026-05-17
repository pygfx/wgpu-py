"""
The entrypoint / script to apply automatic patches to the code.
See README.md for more information.
"""

import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser(description="Run code generation and patching.")
parser.add_argument("--skip-api", action="store_false", dest="api", help="Skip API patching")
parser.add_argument("--skip-wgpu-native", action="store_false", dest="wgpu_native", help="Skip wgpu-native backend updates")
# TODO: this is highly confusing to be swawpped around, but I didn't want to break the existing functionality.
# maybe better to have a --skip a,b,c and --only e,f,g kinda CLI with all default?
parser.add_argument("--js", action="store_true", dest="js", help="Generate JS backend updates")


# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


from codegen import main, file_cache


if __name__ == "__main__":
    # maybe have args here to just run the js writer?
    args = parser.parse_args()
    main(do_api=args.api, do_wgpu_native=args.wgpu_native, do_js=args.js)
    # TODO: we need to actually patch the codegen report if we skip patching some parts of it...
    file_cache.write_changed_files_to_disk()
