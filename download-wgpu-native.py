import argparse
import os
import requests
import sys
import tempfile
from zipfile import ZipFile


RESOURCE_DIR = os.path.join("wgpu", "resources")
VERSION_FILE = os.path.join(RESOURCE_DIR, "wgpu_native-version")


def get_current_version():
    with open(VERSION_FILE, mode="r") as fh:
        return fh.read()


def write_current_version(version):
    with open(VERSION_FILE, mode="w") as fh:
        return fh.write(version)


def download_file(url, filename):
    CHUNK_SIZE = 1024 * 128
    resp = requests.get(url, stream=True)
    with open(filename, mode="wb") as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)


def extract_files(zip_filename, members, path, missing_ok=True):
    z = ZipFile(zip_filename)
    for member in members:
        try:
            z.extract(member, path=path)
        except KeyError:
            if not missing_ok:
                raise


def get_os_string():
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("darwin"):
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"


def main(version, debug, os_string, upstream):
    debug = "debug" if debug else "release"
    filename = f"wgpu-{debug}-{os_string}-{version}.zip"
    url = f"https://github.com/{upstream}/releases/download/{version}/{filename}"
    tmp = tempfile.gettempdir()
    zip_filename = os.path.join(tmp, filename)
    print(f"Downloading {url} to {zip_filename}")
    download_file(url, zip_filename)
    members = [
        "wgpu.h",
        "libwgpu_native.so",
        "libwgpu_native.dylib",
        "wgpu_native.dll",
        "commit-sha",
    ]
    print(f"Extracting {members} to {RESOURCE_DIR}")
    extract_files(zip_filename, members, RESOURCE_DIR)
    current_version = get_current_version()
    if version != current_version:
        print(f"Version changed, updating {VERSION_FILE}")
        write_current_version(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download wgpu-native binaries and " "headers from github releases"
    )
    version = get_current_version()
    parser.add_argument(
        "--version", help=f"Version to download (default: {version})", default=version,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Download debug build " "instead of release build",
        default=False,
    )
    os_string = get_os_string()
    parser.add_argument(
        "--os",
        help=f"Platform to download for (default: {os_string})",
        default=os_string,
    )
    upstream = "Korijn/wgpu"
    parser.add_argument(
        "--upstream",
        help=f"Upstream repository to download release from (default: {upstream})",
        default=upstream,
    )
    args = parser.parse_args()

    main(args.version, args.debug, args.os, args.upstream)
