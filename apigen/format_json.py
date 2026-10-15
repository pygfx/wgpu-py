import os

from apigen_utils import DEFINITIONS_DIR, read_definition_file, write_definition_file


def format_json():
    """Format the static json files for consistency."""
    files = [
        fname
        for fname in os.listdir(DEFINITIONS_DIR)
        if fname.endswith(".json") and fname.startswith(("diff_", "docs_"))
    ]
    for fname in files:
        s = read_definition_file(fname)
        write_definition_file(s, fname)


if __name__ == "__main__":
    format_json()
