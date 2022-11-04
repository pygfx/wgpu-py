"""
The entrypoint / script to apply automatic patches to the code.
See README.md for more information.
"""

import os
import sys


# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


from codegen import main, file_cache  # noqa: E402


if __name__ == "__main__":
    main()
    file_cache.write_changed_files_to_disk()
