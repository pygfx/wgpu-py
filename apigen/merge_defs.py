import json

from apigen_utils import dict_deep_merge
from apigen_defs import read_definition_file, write_definition_file


def merge_defs():
    """Merge the structure definitions into a single definition for the base API."""

    # Read
    webgpu = read_definition_file("webgpu.toml")
    diff_base = read_definition_file("diff_base.toml")
    diff_docs = read_definition_file("diff_docs.toml")

    # Combine
    api = dict_deep_merge(webgpu, diff_base)
    api = dict_deep_merge(api, diff_docs, allow_add=False)

    add_sync_properties_and_methods(api)

    api["meta"] = {
        "description": "Definition of the base wgpu API, derived from 'webgpu.toml', 'diff_base.toml', 'diff_docs.toml'.",
        "notes": "The full API, including docs, in one file. This does *not* include enums/flags/structs from any of the backends.",
    }

    write_definition_file(api, "wgpu_api.toml")
    write_definition_file(api, "wgpu_api.json")


def add_sync_properties_and_methods(api: dict):
    """For every async method, add a sync variant."""
    for _class_name, class_dict in api["classes"].items():
        current_prop_names = set(class_dict["properties"].keys())
        new_properties = {}
        for prop_name, prop_dict in class_dict["properties"].items():
            new_properties[prop_name] = prop_dict
            _make_prop_or_method_async(
                prop_name, prop_dict, current_prop_names, new_properties
            )
        class_dict["properties"] = new_properties

    for _class_name, class_dict in api["classes"].items():
        current_method_names = set(class_dict["methods"].keys())
        new_methods = {}
        for method_name, method_dict in class_dict["methods"].items():
            _make_prop_or_method_async(
                method_name, method_dict, current_method_names, new_methods
            )
        class_dict["methods"] = new_methods


def _make_prop_or_method_async(the_name, the_dict, current_names, new_collection):
    # Add the current dict as-is
    new_collection[the_name] = the_dict
    # Need to replace?
    type_str = the_dict["type"] if "type" in the_dict else the_dict["returns"]["type"]
    if type_str.startswith(("GPUPromise[", "Promise[")):
        # Establish base name
        base_name = the_name
        if the_name.lower().endswith("async"):
            base_name = base_name[:-5].rstrip("_")
            if base_name in current_names:
                return  # no need to replicate, e.g. create_compute_pipeline
        # Create alt sync version
        sync_type = type_str.split("Promise[")[1].rstrip("]")
        sync_dict = json.loads(json.dumps(the_dict))
        if "type" in sync_dict:
            sync_dict["type"] = sync_type
        else:
            sync_dict["returns"]["type"] = sync_type
            sync_dict["returns"]["doc"] = ""
        sync_dict["doc"] = f"The synchronous variant of {base_name}_async()."
        # Replace
        new_collection.pop(the_name)
        new_collection[base_name + "_async"] = the_dict
        new_collection[base_name + "_sync"] = sync_dict


if __name__ == "__main__":
    merge_defs()
