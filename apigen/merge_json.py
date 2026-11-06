import json

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

    add_sync_methods(api)

    api["meta"] = {
        "filename": "wgpu_api.json",
        "description": "Definition of the base wgpu API, derived from 'webgpu.json', 'diff_base.json', 'diff_docs.json'.",
        "notes": "The full API, including docs, in one file. This does *not* include enums/flags/structs from any of the backends.",
    }

    write_definition_file(api, "wgpu_api.json")


def add_sync_methods(api: dict):
    """For every async method, add a sync variant."""
    for _class_name, class_dict in api["classes"].items():
        new_methods = {}
        for method_name, method_dict in class_dict["methods"].items():
            new_methods[method_name] = method_dict
            type_str = method_dict["returns"]["type"]
            if type_str.startswith(("GPUPromise[", "Promise[")):
                base_name = method_name
                if method_name.lower().endswith("async"):
                    base_name = base_name[:-5].rstrip("_")
                    if base_name in class_dict["methods"]:
                        continue  # no need to replicate, e.g. create_compute_pipeline
                sync_dict = json.loads(json.dumps(method_dict))
                sync_dict["returns"]["type"] = type_str.split("Promise[")[1].rstrip("]")
                sync_dict["returns"]["doc"] = ""
                sync_dict["doc"] = f"The synchronous variant of {base_name}_async()."
                new_methods.pop(method_name)
                new_methods[base_name + "_async"] = method_dict
                new_methods[base_name + "_sync"] = sync_dict

        class_dict["methods"] = new_methods


if __name__ == "__main__":
    merge_json()
