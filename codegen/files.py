"""
Simple utilities to handle files, including a mini virtual file system.
"""

import os


lib_dir = os.path.abspath(os.path.join(__file__, "..", "..", "wgpu"))


def read_file(*fname):
    """Read a file from disk using the relative filename. Line endings are normalized."""
    filename = os.path.join(lib_dir, *fname)
    with open(filename, "rb") as f:
        return f.read().decode().replace("\r\n", "\n").replace("\r", "\n")


class FileCache:
    """An in-memory file cache, to allow performing the codegen
    in-memory, providing checks on what is actually changed, enabling
    dry runs for tests, and make it easier to write back files with the
    correct line endings.
    """

    _filenames_to_change = [
        "_classes.py",
        "flags.py",
        "enums.py",
        "structs.py",
        "backends/wgpu_native/_api.py",
        "backends/wgpu_native/_mappings.py",
        "resources/codegen_report.md",
    ]

    def __init__(self):
        self._file_contents = {}
        self._files_written = set()

    def reset(self):
        """Reset the cache, populating the files with a copy from disk."""
        self._file_contents.clear()
        for fname in self.filenames_to_change:
            self.write(fname, read_file(fname))
        self._files_written.clear()

    @property
    def filenames_to_change(self):
        """The (relative) filenames that the codegen is allowed to change."""
        return tuple(self._filenames_to_change)

    @property
    def filenames_written(self):
        """The (relative) filenames that are actually written."""
        return set(self._files_written)

    def write(self, fname, text):
        """Write to a (virtual) file. The text is a string with LF newlines."""
        assert fname in self.filenames_to_change
        self._files_written.add(fname)
        self._file_contents[fname] = text

    def read(self, fname):
        """Read from a (virtual) file. Returns text with LF newlines."""
        assert fname in self.filenames_to_change
        return self._file_contents[fname]

    def write_changed_files_to_disk(self):
        """Write the virtual files to disk, using appropriate newlines."""
        # Get reference line ending chars
        with open(os.path.join(lib_dir, "__init__.py"), "rb") as f:
            text = f.read().decode()
        line_endings = get_line_endings(text)
        # Write files
        for fname in self.filenames_to_change:
            text = self.read(fname)
            filename = os.path.join(lib_dir, fname)
            with open(filename, "wb") as f:
                f.write(text.replace("\n", line_endings).encode())


file_cache = FileCache()


def get_line_endings(text):
    """Detect whether the line endings in use is CR LF or CRLF."""
    # Count how many line ending chars there are
    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    cr_count = text.count("\r") - crlf_count
    assert lf_count + cr_count + crlf_count >= 4
    # Check what's used the most, or whether it's a combination.
    if lf_count > cr_count and lf_count > crlf_count:
        return "\n"
    elif cr_count > lf_count and cr_count > crlf_count:
        return "\r"
    else:
        return "\r\n"
