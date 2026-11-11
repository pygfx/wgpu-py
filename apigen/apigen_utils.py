import sys
import subprocess
from pathlib import Path

DEFINITIONS_DIR = Path(__file__).parent.absolute() / "definitions"
ROOT_DIR = Path(__file__).parents[1].absolute()


def read_file(*fname):
    """Read a file from disk using the relative filename. Line endings are normalized."""
    filename = ROOT_DIR.joinpath(*fname)
    with open(filename, "rb") as f:
        return (
            f.read()
            .decode()
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\\\n", "")  # Macros escaping a newline
        )


def write_file(text: str, *fname):
    """Write the given text to disk, using the relative filename. Line endings are matched to what's used for __init__.py."""
    filename = ROOT_DIR.joinpath(*fname)
    if not text.endswith("\n"):
        text = text + "\n"
    with open(filename, "wb") as f:
        f.write(text.replace("\n", LE).encode())


def _get_line_endings(text):
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


LE = _get_line_endings((ROOT_DIR / "apigen" / "__init__.py").read_bytes().decode())


def dict_deep_merge(a: dict, b: dict, allow_add: bool = True) -> dict:
    """Return a new dict: b merged into a, recursing into nested dicts."""
    result = dict(a)
    for key, val in b.items():
        if isinstance(val, dict):
            if key in result and isinstance(result[key], dict):
                if "diff" in val:
                    assert val["diff"] == "REMOVE"
                    result.pop(key, None)
                else:
                    result[key] = dict_deep_merge(result[key], val, allow_add)
            elif allow_add:
                result[key] = val
        else:
            result[key] = val

    return result


def indent_code(text: str, indent=4) -> str:
    text = text.strip()
    lines = []
    for line in text.split("\n"):
        lines.append(" " * indent + line.rstrip())
    lines.append("")
    return "\n".join(lines)


class FormatError(Exception):
    pass


def format_code(src, singleline=False):
    """Format the given src string. If singleline is True, all function
    signatures become single-line, so they can be parsed and updated.
    """
    line_length = 320 if singleline else 88
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--line-length",
        str(line_length),
        "--no-cache",
        "--stdin-filename",
        "tmp.py",
    ]

    p = subprocess.run(cmd, input=src.encode(), capture_output=True)

    if p.returncode:
        stdout = p.stdout.decode(errors="ignore")
        stderr = p.stderr.decode(errors="ignore")
        raise FormatError(
            f"Could not format source ({p.returncode}).\n\nstdout: "
            + stdout
            + "\n\nstderr: "
            + stderr
        )
    else:
        result = p.stdout.decode(errors="ignore")

    # Make defs single-line. You'd think that setting the line length
    # to a very high number would do the trick, but it does not.
    if singleline:
        result = _make_sigs_singleline(result)

    return result


def _make_sigs_singleline(code):
    lines1 = code.splitlines()
    lines2 = []

    sig_state = 0
    sig_brace_depth = 0
    sig_line = ""
    sig_comment = ""

    for line in lines1:
        # Check to enter in signature-retrieval mode
        if not sig_state:
            if line.lstrip().startswith(("def ", "async def", "class ")):
                sig_state = 1
                sig_line = ""
                sig_comment = ""
                sig_brace_depth = 0
                if line.lstrip().startswith("class ") and "(" not in line:
                    sig_state = 3  # search for closing colon directly
            else:
                lines2.append(line)
                continue

        # If we get here, we're in a signature

        # Handle comment
        line, _, c = line.partition("#")
        line = line.rstrip()
        c = c.strip()
        if c:
            sig_comment += " " + c.strip()

        if sig_state == 1:
            # Find the first opening brace
            i = line.find("(")
            if i >= 0:
                i += 1
                sig_brace_depth = 1
                sig_line += line[:i]
                line = line[i:]
                sig_state = 2

        line = line.lstrip()
        if sig_line.endswith(","):
            line = " " + line

        if sig_state == 2:
            # Resolve braces until we find the closing brace
            while True:
                i1 = line.find("(")
                i2 = line.find(")")
                if i1 >= 0 and i1 < i2:
                    i = i1 + 1
                    sig_brace_depth += 1
                    sig_line += line[:i]
                    line = line[i:]
                elif i2 >= 0:
                    i = i2 + 1
                    sig_brace_depth -= 1
                    sig_line += line[:i]
                    line = line[i:]
                else:
                    break
                if sig_brace_depth == 0:
                    sig_state = 3
                    break

        if sig_state == 3:
            # Find the closing colon
            i = line.find(":")
            if i >= 0:
                i += 1
                sig_line += line[:i]
                line = line[i:]
                # Finish the signature line
                sig_line = sig_line.replace(", )", ")")
                if sig_comment:
                    sig_line += "  #" + sig_comment
                lines2.append(sig_line)
                if line.strip():
                    lines2.append(" " * 12 + line.strip())
                # End the search
                sig_state = 0
                line = ""

        sig_line += line

    lines2.append("")
    return "\n".join(lines2)
