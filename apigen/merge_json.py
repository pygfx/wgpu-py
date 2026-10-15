from apigen_utils import dict_deep_merge, read_definition_file, write_definition_file


def merge_json():
    """Merge the structure definitions into a single definition for the base API."""

    # Read
    api_web = read_definition_file("wgpu_api_web.json")
    diff_base = read_definition_file("diff_base.json")
    docs_ai = read_definition_file("docs_ai.json")
    docs_dev = read_definition_file("docs_dev.json")

    # Combine
    api = dict_deep_merge(api_web, diff_base)

    # Insert docs
    api = dict_deep_merge(api, docs_ai, allow_add=False)
    api = dict_deep_merge(api, docs_dev, allow_add=False)

    api["meta"] = {
        "filename": "wgpu_api.json",
        "description": "Definition of the base wgpu API, derived from 'wgpu_api_web.json', 'diff_base.json', 'docs_ai.json', 'docs_dev.json'.",
        "notes": "The full API, including docs, in one file. This does *not* include enums/flags/structs from any of the backends.",
    }

    write_definition_file(api, "wgpu_api.json")


if __name__ == "__main__":
    merge_json()
