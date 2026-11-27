"""
The entrypoint / script to apply automatic patches to the code.
See README.md for more information.
"""

import io
import os
import sys
from argparse import ArgumentParser

from codegen import actions
from codegen_utils import print, PrintToFile, write_file


parser = ArgumentParser(description="codegen - run code generation and patching.")

for func in actions:
    parser.add_argument(
        "--" + func.__name__.replace("_", "-"),
        action="store_true",
        dest=func.__name__,
        help=func.__doc__.strip().splitlines()[0],
    )

parser.add_argument(
    "--all",
    action="store_true",
    dest="all",
    help="Run all steps of the codegen (except download).",
)

parser.add_argument(
    "--doc",
    action="store",
    dest="doc",
    help="Show doc on specific action.",
)

# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


def run_all():
    log = io.StringIO()
    with PrintToFile(log):
        print("# Code generation report")
        for func in actions:
            if "download" in func.__name__:
                continue
            print(f"\n## {func.__name__}")
            func()
        write_file(log.getvalue(), "codegen", "definitions", "codegen_report.md")


if __name__ == "__main__":
    args = parser.parse_args()
    enabled = [k for k, v in vars(args).items() if v is True]

    if args.doc is not None:
        the_action = args.doc.replace("-", "_").strip("_")
        for func in actions:
            if func.__name__ == the_action:
                print(f"--- More info for {func.__name__}() ---\n")
                print(func.__doc__)
                break
        else:
            sys.exit(f"Unknown action {the_action!r}")
    elif not enabled:
        parser.print_help()
    elif "all" in enabled:
        run_all()
    else:
        for name in enabled:
            for func in actions:
                if name == func.__name__:
                    print(f"## {func.__name__}")
                    func()
