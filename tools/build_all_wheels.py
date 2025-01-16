"""
Script to build all wheels.

Can be run from any Unix machine.
Relies on the build hook in `hatch_build.py` consuming WGPU_BUILD_PLATFORM_INFO.
"""

import os
import sys
import shutil
import hashlib
import zipfile
from subprocess import run

from download_wgpu_native import main as download_lib


# Define platform tags. These are pairs of wgpu-native-tag and wheel-tag. The
# former is used to download the appropriate binary lib, the latter is used to
# give the resulting wheel the correct name. Note that the build system does
# not check the name, so a typo here results in a broken wheel.
#
# Linux: The wgpu-native wheels are build on manylinux. The manylinux version
# listed below must match that.
#
# MacOS: On wgpu-native the MACOSX_DEPLOYMENT_TARGET is set to 10.13, I presume
# this affects the macos version in the name below.
#
# TODO: To help automate the naming, wgpu-native could include a file with
# build-info in the archive. This'd include the os and architecture themselves,
# the version of manylinux, and macos deployment target.

PLATFORM_TAGS = [
    ("windows_x86_64", "win_amd64"),
    ("windows_aarch64", "win_arm64"),
    ("windows_i686", "win32"),
    ("macos_x86_64", "macosx_10_9_x86_64"),
    ("macos_aarch64", "macosx_11_0_arm64"),
    ("linux_x86_64", "manylinux_2_28_x86_64"),
    ("linux_aarch64", "manylinux_2_28_aarch64"),
]


# --- Prepare


if not sys.platform.startswith(("darwin", "win")):
    print("WARNING: Building releases only really works on Unix")

# Make sure we're in the project root, no matter where this is alled from.
root_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
os.chdir(root_dir)
print(os.getcwd())

# Remove the dist directory for a fresh start
if os.path.isdir("dist"):
    shutil.rmtree("dist")


# --- Build


# Build all wheels
for platform_info in PLATFORM_TAGS:
    os.environ["WGPU_BUILD_PLATFORM_INFO"] = " ".join(platform_info)
    run([sys.executable, "-m", "build", "-n", "-w"])

# Build sdist
run([sys.executable, "-m", "build", "-n", "-s"])

# Restore wgpu-native
download_lib()


# --- Checks produced files


all_tags = set(platform_info[1] for platform_info in PLATFORM_TAGS)
assert len(all_tags) == len(PLATFORM_TAGS), "Wheel tags in PLATFORM_TAGS are not unique"

found_files = os.listdir("dist")
found_wheels = [fname for fname in found_files if fname.endswith(".whl")]
found_tags = {fname.split("none-")[1].split(".")[0] for fname in found_wheels}
assert found_tags == all_tags, (
    f"Found tags does not match expected tags: {found_tags}\n{all_tags}"
)

found_others = list(set(found_files) - set(found_wheels))
assert len(found_others) == 1 and found_others[0].endswith(".tar.gz"), (
    f"Found unexpected files: {found_others}"
)

for archive_name in found_wheels:
    assert "-any-" not in archive_name, (
        f"There should not be an 'any' wheel: {archive_name}"
    )


# --- Report and check content of archives


print("Dist archives:")

# Simple check for sdist archive
for archive_name in found_others:
    size = os.stat("dist/" + archive_name).st_size
    print(f"{archive_name}  ({size / 1e6:0.2f} MB)")
    assert size < 1e6, f"Did not expected {archive_name} to be this large"

# Collect content of each wheel
hash_to_file = {}
for archive_name in found_wheels:
    size = os.stat("dist/" + archive_name).st_size
    print(f"{archive_name}  ({size / 1e6:0.2f} MB)")
    z = zipfile.ZipFile("dist/" + archive_name)
    flat_map = {os.path.basename(fi.filename): fi.filename for fi in z.filelist}
    lib_hashes = []
    for fname in flat_map:
        if fname.endswith((".so", ".dll", ".dylib")):
            bb = z.read(flat_map[fname])
            hash = hashlib.sha256(bb).hexdigest()
            lib_hashes.append(hash)
            print(f"    - {fname}  ({len(bb) / 1e6:0.2f} MB)\n      {hash}")
    assert len(lib_hashes) == 1, (
        f"Expected 1 lib per wheel, got {len(lib_hashes)} in {archive_name}"
    )
    hash = lib_hashes[0]
    assert hash not in hash_to_file, (
        f"Same lib found in {hash_to_file[hash]} and archive_name"
    )
    hash_to_file[hash] = archive_name

# Meta check
assert set(hash_to_file.values()) == set(found_wheels)
