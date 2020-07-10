import argparse
import os
import re
import requests
import sys
import tempfile
from zipfile import ZipFile


# The directory containing non-python resources that are included in packaging
RESOURCE_DIR = os.path.join("wgpu", "resources")
# The version installed through this script is tracked in the backend module
VERSION_FILE = os.path.join("wgpu", "backends", "rs.py")

# Whether to ensure we export \n instead of \r\n
FORCE_SIMPLE_NEWLINES = False
if sys.platform.startswith("win"):
    sample = open(os.path.join(RESOURCE_DIR, "codegen_report.md"), "rb").read()
    if sample.count(b"\r\n") == 0:
        FORCE_SIMPLE_NEWLINES = True


def get_current_version():
    with open(VERSION_FILE) as fh:
        return re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


def write_current_version(version, commit_sha):
    with open(VERSION_FILE, "rb") as fh:
        file_content = fh.read().decode()
    file_content = re.sub(
        r"__version__ = \".*?\"", f'__version__ = "{version}"', file_content,
    )
    file_content = re.sub(
        r"__commit_sha__ = \".*?\"", f'__commit_sha__ = "{commit_sha}"', file_content,
    )
    with open(VERSION_FILE, mode="wb") as fh:
        fh.write(file_content.encode())


def download_file(url, filename):
    resp = requests.get(url, stream=True)
    with open(filename, mode="wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 128):
            fh.write(chunk)


def extract_file(zip_filename, member, path):
    z = ZipFile(zip_filename)
    os.makedirs(path, exist_ok=True)
    z.extract(member, path=path)
    if member.endswith(".h") and FORCE_SIMPLE_NEWLINES:
        filename = os.path.join(path, member)
        bb = open(filename, "rb").read()
        with open(filename, "wb") as f:
            f.write(bb.replace(b"\r\n", b"\n"))


def get_os_string():
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("darwin"):
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    else:
        # We do not provide binaries for this platform. Note that we can
        # have false positives, e.g. on ARM Linux. We assume that users on
        # such platforms are aware and arrange for the wgpu lib themselves.
        raise RuntimeError(f"Platform '{sys.platform}' not supported")


def get_arch():
    return "64" if sys.maxsize > 2 ** 32 else "32"  # True on 64-bit Python interpreters


def main(version, os_string, arch, upstream):
    for build in ("release", "debug"):
        filename = f"wgpu-{os_string}-{arch}-{build}.zip"
        url = f"https://github.com/{upstream}/releases/download/v{version}/{filename}"
        tmp = tempfile.gettempdir()
        zip_filename = os.path.join(tmp, filename)
        print(f"Downloading {url} to {zip_filename}")
        download_file(url, zip_filename)
        headerfile = "wgpu.h"
        binaryfile = None
        if os_string == "linux":
            binaryfile = "libwgpu_native.so"
        elif os_string == "macos":
            binaryfile = "libwgpu_native.dylib"
        elif os_string == "windows":
            binaryfile = "wgpu_native.dll"
        else:
            raise RuntimeError(f"Platform '{os_string}' not supported")
        root, ext = os.path.splitext(binaryfile)
        binaryfile_name = root + "-" + build + ext
        print(f"Extracting {headerfile} to {RESOURCE_DIR}")
        extract_file(zip_filename, headerfile, RESOURCE_DIR)
        print(f"Extracting {binaryfile} to {RESOURCE_DIR}")
        extract_file(zip_filename, binaryfile, RESOURCE_DIR)
        os.rename(
            os.path.join(RESOURCE_DIR, binaryfile),
            os.path.join(RESOURCE_DIR, binaryfile_name),
        )
    current_version = get_current_version()
    if version != current_version:
        print(f"Version changed, updating {VERSION_FILE}")
        filename = "commit-sha"
        url = f"https://github.com/{upstream}/releases/download/v{version}/{filename}"
        commit_sha_filename = os.path.join(tmp, filename)
        print(f"Downloading {url} to {commit_sha_filename}")
        download_file(url, commit_sha_filename)
        with open(commit_sha_filename) as fh:
            commit_sha = fh.read().strip()
        write_current_version(version, commit_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download wgpu-native binaries and headers from github releases"
    )
    version = get_current_version()
    parser.add_argument(
        "--version", help=f"Version to download (default: {version})", default=version
    )
    os_string = get_os_string()
    parser.add_argument(
        "--os",
        help=f"Platform to download for (default: {os_string})",
        default=os_string,
        choices=("linux", "macos", "windows"),
    )
    arch_string = get_arch()
    parser.add_argument(
        "--arch",
        help=f"Architecture to download for (default: {arch_string})",
        default=arch_string,
        choices=("32", "64"),
    )
    upstream = "gfx-rs/wgpu-native"
    parser.add_argument(
        "--upstream",
        help=f"Upstream repository to download release from (default: {upstream})",
        default=upstream,
    )
    args = parser.parse_args()

    main(args.version, args.os, args.arch, args.upstream)
