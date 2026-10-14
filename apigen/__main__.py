"""
The entrypoint / script to apply automatic patches to the code.
See README.md for more information.
"""

import os
import sys
from argparse import ArgumentParser


parser = ArgumentParser(description="Run code generation and patching.")

parser.add_argument(
    "--download-idl",
    action="store_true",
    dest="download_idl",
    help="Download the latest webgpu.idl",
)


# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


from apigen import download_idl


if __name__ == "__main__":
    args = parser.parse_args()
    if args.download_idl:
        print("Hey")
        download_idl()
    elif False:
        pass
        # main(do_api=args.api, do_wgpu_native=args.wgpu_native, do_js=args.js)
        # main()
        # file_cache.write_changed_files_to_disk()
    else:
        parser.print_help()
        sys.exit(0)
