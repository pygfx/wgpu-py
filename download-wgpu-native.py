import argparse
import os
import requests
import sys
import tempfile
from zipfile import ZipFile


# The directory containing non-python resources that are included in packaging
RESOURCE_DIR = os.path.join("wgpu", "resources")
# A text file used to track the version installed through this script
VERSION_FILE = os.path.join(RESOURCE_DIR, "wgpu_native-version")


def get_current_version():
    with open(VERSION_FILE, mode="r") as fh:
        return fh.read().strip()


def write_current_version(version):
    with open(VERSION_FILE, mode="w") as fh:
        return fh.write(version.strip())


def download_file(url, filename):
    CHUNK_SIZE = 1024 * 128
    resp = requests.get(url, stream=True)
    with open(filename, mode="wb") as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)


def extract_files(zip_filename, members, path):
    z = ZipFile(zip_filename)
    for member in members:
        z.extract(member, path=path)


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
    filename = f"wgpu-{os_string}-{arch}-release.zip"
    url = f"https://github.com/{upstream}/releases/download/{version}/{filename}"
    tmp = tempfile.gettempdir()
    zip_filename = os.path.join(tmp, filename)
    print(f"Downloading {url} to {zip_filename}")
    download_file(url, zip_filename)
    members = ["wgpu.h", "commit-sha"]
    if os_string == "linux":
        members.append("libwgpu_native.so")
    elif os_string == "macos":
        members.append("libwgpu_native.dylib")
    elif os_string == "windows":
        members.append("wgpu_native.dll")
    else:
        raise RuntimeError(f"Platform '{os_string}' not supported")
    print(f"Extracting {members} to {RESOURCE_DIR}")
    extract_files(zip_filename, members, RESOURCE_DIR)
    current_version = get_current_version()
    if version != current_version:
        print(f"Version changed, updating {VERSION_FILE}")
        write_current_version(version)


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
    upstream = "Korijn/wgpu-bin"
    parser.add_argument(
        "--upstream",
        help=f"Upstream repository to download release from (default: {upstream})",
        default=upstream,
    )
    args = parser.parse_args()

    main(args.version, args.os, args.arch, args.upstream)
