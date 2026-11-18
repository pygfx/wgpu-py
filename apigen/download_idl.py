import urllib.request

from apigen_utils import DEFINITIONS_DIR


def download_idl():
    """Download the latest webgpu.idl from the gpuweb repo."""

    url = (
        "https://raw.githubusercontent.com/gpuweb/gpuweb/refs/heads/gh-pages/webgpu.idl"
    )

    with urllib.request.urlopen(url, timeout=10) as response:
        data = response.read()

    filename = DEFINITIONS_DIR / "webgpu.idl"

    with open(filename, "wb") as f:
        f.write(data)

    print(f"Wrote to {filename}")


if __name__ == "__main__":
    download_idl()
