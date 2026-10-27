from apigen_utils import dict_deep_merge, read_definition_file, write_definition_file


def merge_json():
    """Merge the structure definitions into a single definition for the base API."""

    # Read
    webgpu = read_definition_file("webgpu.json")
    diff_base = read_definition_file("diff_base.json")
    diff_docs = read_definition_file("diff_docs.json")

    # Combine
    api = dict_deep_merge(webgpu, diff_base)
    api = dict_deep_merge(api, diff_docs, allow_add=False)

    api["meta"] = {
        "filename": "wgpu_api.json",
        "description": "Definition of the base wgpu API, derived from 'webgpu.json', 'diff_base.json', 'diff_docs.json'.",
        "notes": "The full API, including docs, in one file. This does *not* include enums/flags/structs from any of the backends.",
    }

    write_definition_file(api, "wgpu_api.json")


if __name__ == "__main__":
    merge_json()
