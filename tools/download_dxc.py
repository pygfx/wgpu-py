# helper script to download files needed to use dxc. Information based on: https://docs.rs/wgpu/25.0.2/wgpu/enum.Dx12Compiler.html#variant.DynamicDxc
# this will improve for v26 so this script should be updated for wgpu 26, let @Vipitis know if this script is outdated.
# inspired by download_wgpu_native.py

import argparse
import os
import tempfile
from zipfile import ZipFile
import requests

from download_wgpu_native import RESOURCE_DIR, download_file, get_os_string


# changed to use a specific arch
def extract_file(zip_filename, member, path):
    # Read file from archive, find it no matter the folder structure
    z = ZipFile(zip_filename)
    bb = z.read(
        f"bin/x64/{member}"
    )  # this one works for me... maybe we can parameterize arch?
    # Write to disk
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, member), "wb") as f:
        f.write(bb)


def get_latest_release() -> str:
    url = "https://api.github.com/repos/microsoft/DirectXShaderCompiler/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get latest release: {response.status_code=}")
    data = response.json()
    return data["tag_name"]  # .removeprefix("v") # to be inline with the other script?


def get_filename(version: str) -> str:
    """returns the filename of the dxc_yyyy_mm_dd.zip file so we can download it"""
    url = f"https://api.github.com/repos/microsoft/DirectXShaderCompiler/releases/tags/{version}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get release info: {response.status_code=}")
    data = response.json()
    for asset in data["assets"]:
        if asset["name"].startswith("dxc_") and asset["name"].endswith(".zip"):
            return asset["name"]
    raise RuntimeError(f"Couldn't find dxc archive for release {version}")


def main(version=None):
    if version is None:
        version = get_latest_release()
    os_string = get_os_string()
    if os_string != "windows":
        raise RuntimeError("Dxc only supported on Windows")
    filename = get_filename(version)
    url = f"https://github.com/microsoft/DirectXShaderCompiler/releases/download/{version}/{filename}"  # or use the api response for "browser_download_url"?
    tmp = tempfile.gettempdir()
    zip_filename = os.path.join(tmp, filename)
    print(f"Downloading {url}")
    download_file(url, zip_filename)
    compiler_file = "dxcompiler.dll"
    print(f"Extracting {compiler_file} to {RESOURCE_DIR}")
    extract_file(zip_filename, compiler_file, RESOURCE_DIR)

    # cleanup of tempfile?
    # os.remove(zip_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Dxc from github release.")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version of dxc to download, defaults to latest.",
    )
    args = parser.parse_args()

    main(version=args.version)
