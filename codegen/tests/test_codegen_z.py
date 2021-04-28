"""
Applying the codegen should not introduce changes.
"""

import os
import time

from codegen.__main__ import main
from codegen.utils import lib_dir


changing_files = {
    os.path.join(lib_dir, "base.py"),
    os.path.join(lib_dir, "flags.py"),
    os.path.join(lib_dir, "enums.py"),
    os.path.join(lib_dir, "structs.py"),
    os.path.join(lib_dir, "backends", "rs.py"),
    os.path.join(lib_dir, "backends", "rs_mappings.py"),
    os.path.join(lib_dir, "resources", "codegen_report.md"),
}


def test_that_code_is_up_to_date():
    """Test that running the codegen updates what we expect, but does not introduce changes."""

    # Obtain mtime of all files
    mtimes = {}
    for root, dirs, files in os.walk(lib_dir):
        if "__pycache__" in root:
            continue
        for fname in files:
            filename = os.path.join(lib_dir, root, fname)
            mtimes[filename] = os.path.getmtime(filename)

    # Give it some time
    time.sleep(0.2)
    ref_time = time.time()
    time.sleep(0.2)

    assert all(t < ref_time for t in mtimes.values())

    # Create cache
    file_cache = {}
    for filename in changing_files:
        assert os.path.isfile(filename)
        with open(filename, "rb") as f:
            file_cache[filename] = f.read()

    # Double-check that mtimes have not changed
    all(t == os.path.getmtime(filename) for filename, t in mtimes.items())

    # Apply codegen
    main()

    # Get files that have changed
    updated = {
        filename for filename, t in mtimes.items() if t != os.path.getmtime(filename)
    }

    # Check that exactly the files updated that we expected
    assert updated == changing_files

    # But they should not have changed
    for filename in changing_files:
        content1 = file_cache[filename]
        with open(filename, "rb") as f:
            content2 = f.read()
        assert content1 == content2
        assert ref_time < os.path.getmtime(filename)

    print("Codegen check ok!")


def test_that_codegen_report_has_no_errors():
    filename = os.path.join(lib_dir, "resources", "codegen_report.md")
    with open(filename, "rb") as f:
        text = f.read().decode()

    # The codegen uses a prefix "ERROR:" for unacceptable things.
    # All caps, some function names may contain the name "error".
    assert "ERROR" not in text


if __name__ == "__main__":
    test_that_code_is_up_to_date()
