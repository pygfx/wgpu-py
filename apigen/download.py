import urllib.request
from apigen_utils import DEFINITIONS_DIR


def download_idl():
    url = (
        "https://raw.githubusercontent.com/gpuweb/gpuweb/refs/heads/gh-pages/webgpu.idl"
    )

    with urllib.request.urlopen(url, timeout=10) as response:
        data = response.read()

    filename = DEFINITIONS_DIR / "webgpu.idl"

    with open(filename, "wb") as f:
        f.write(data)

    print(f"Wrote to {filename}")


def download_ts():
    # Typescript type definition with docs
    url = (
        "https://raw.githubusercontent.com/gpuweb/types/refs/heads/main/dist/index.d.ts"
    )

    with urllib.request.urlopen(url, timeout=10) as response:
        data = response.read()

    filename = DEFINITIONS_DIR / "index.d.ts"

    with open(filename, "wb") as f:
        f.write(data)

    print(f"Wrote to {filename}")


if __name__ == "__main__":
    # download_idl()
    download_ts()
