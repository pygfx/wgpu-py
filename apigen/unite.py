"""Unite/merge the structure definitions into more final definitions. """

from apigen_utils import read_definition_file, dict_deep_merge, DEFINITIONS_DIR


def unite_toml_files():
    unite_base_api()
    unite_docs()


def unite_base_api():

    # Read
    api_web = read_definition_file("wgpu_api_web.toml")
    diff_base = read_definition_file("diff_base.toml")

    # Combine
    api = dict_deep_merge(api_web, diff_base)

    # Generate toml
    # We generate the whole structure, but ignore "do"" fields, assuming they are handled separately.

    fname = "wgpu_api.toml"
    hline = "# " + '-' * 80
    hline3 = "\n" + hline + "\n" + hline + "\n" + hline

    toml_lines = [
        f"# {fname}: the base wgpu-py API, generated from 'wgpu_api_web.toml' and 'diff_base.toml'."
    ]

    toml_lines.append(hline3)
    toml_lines.append("[flags]")
    for flag_name, flag_dict in api["flags"].items():
        toml_lines.append(f"\n[flags.{flag_name}]")
        for k, d in flag_dict['fields'].items():
            toml_lines.append(f"fields.{k} = {{ value = {d['value']!r} }}")

    toml_lines.append(hline3)
    toml_lines.append("[enums]")
    for enum_name, enum_dict in api["enums"].items():
        toml_lines.append(f"\n[enums.{enum_name}]")
        for k, d in enum_dict['fields'].items():
            toml_lines.append(f"fields.{k} = {{ value = '{d['value']}' }}")

    toml_lines.append(hline3)
    toml_lines.append("[structs]")
    for struct_name, struct_dict in api["structs"].items():
        toml_lines.append(f"\n[structs.{struct_name}]")
        for k, d in struct_dict.get('fields', {}).items():
            toml_lines.append(f"fields.{k} = {{ type = {d['type']!r}, default = '{d['default']}', required = {repr(d['required']).lower()} }}")

    toml_lines.append(hline3)
    toml_lines.append("[classes]")
    for class_name, class_dict in api["classes"].items():
        toml_lines.append(f"\n[classes.{class_name}]")
        toml_lines.append(f"bases = {class_dict['bases']!r}")
        for prop_name, d in class_dict.get('properties', {}).items():
            toml_lines.append(f"properties.{prop_name} = {{ type = {d['type']!r} }}")
        toml_lines.append(f"[classes.{class_name}.methods]")
        for method_name, method_dict in class_dict['methods'].items():
            toml_lines.append(f"[classes.{class_name}.methods.{method_name}]")
            toml_lines.append(f"returns = {{ type = '{method_dict['returns']['type']}' }}")
            for arg_name, d in method_dict.get('args', {}).items():
                toml_lines.append(f"args.{arg_name} = {{ type = {d['type']!r}, default = '{d['default']}', required = {repr(d['required']).lower()} }}")

    # Write

    toml_lines.append("")
    toml = "\n".join(toml_lines)

    with open(DEFINITIONS_DIR / fname, "wb") as f:
        f.write(toml.encode())

    # Check that file is correct toml and structure
    read_definition_file(fname)


def unite_docs():

    # Read definitions
    api = read_definition_file("wgpu_api.toml")
    docs_ai = read_definition_file("docs_ai.toml")
    docs_ts = read_definition_file("docs_ts.toml")
    docs_dev = read_definition_file("docs_dev.toml")

    # Combine into api. Only add docs for stuff that's actually in the API
    docs = api
    docs = dict_deep_merge(docs, docs_ai, allow_add=False)
    docs = dict_deep_merge(docs, docs_ts, allow_add=False)
    docs = dict_deep_merge(docs, docs_dev, allow_add=False)

    # Generate toml, only the doc fields. We assume that there is a doc
    # field for every leaf node. If not, add it to docs_dev.toml!

    fname = "wgpu_docs.toml"
    hline = "# " + '-' * 80
    hline3 = "\n" + hline + "\n" + hline + "\n" + hline

    toml_lines = [
         f"# {fname}",
         "# Documentation for the wgpu-py API, composed from 'docs_ai.toml', 'docs_ts.toml', and 'docs_dev.toml'",
    ]

    toml_lines.append(hline3)
    toml_lines.append("[flags]")
    for flag_name, flag_dict in docs["flags"].items():
        toml_lines.append(f"\n[flags.{flag_name}]")
        toml_lines.append(f'doc = """{flag_dict["doc"]}"""')
        for k, d in flag_dict['fields'].items():
            toml_lines.append(f'fields.{k}.doc = """{d["doc"]}"""')

    toml_lines.append(hline3)
    toml_lines.append("[enums]")
    for enum_name, enum_dict in docs["enums"].items():
        toml_lines.append(f"\n[enums.{enum_name}]")
        toml_lines.append(f'doc = """{enum_dict["doc"]}"""')
        for k, d in enum_dict['fields'].items():
            toml_lines.append(f'fields.{k}.doc = """{d["doc"]}"""')

    toml_lines.append(hline3)
    toml_lines.append("[structs]")
    for struct_name, struct_dict in docs["structs"].items():
        toml_lines.append(f"\n[structs.{struct_name}]")
        toml_lines.append(f'doc = """{struct_dict["doc"]}"""')
        for k, d in struct_dict.get('fields', {}).items():
            toml_lines.append(f'fields.{k}.doc = """{d["doc"]}"""')

    toml_lines.append(hline3)
    toml_lines.append("[classes]")
    for class_name, class_dict in docs["classes"].items():
        toml_lines.append(f"\n[classes.{class_name}]")
        toml_lines.append(f'doc = """{class_dict["doc"]}"""')
        for prop_name, d in class_dict.get('properties', {}).items():
            toml_lines.append(f'properties.{prop_name}.doc = """{d["doc"]}"""')
        toml_lines.append(f"[classes.{class_name}.methods]")
        for method_name, method_dict in class_dict['methods'].items():
            toml_lines.append(f'[classes.{class_name}.methods.{method_name}]')
            toml_lines.append(f'doc = """{method_dict["doc"]}"""')
            toml_lines.append(f'returns.doc = """{method_dict["returns"]["doc"]}"""')
            for arg_name, d in method_dict.get('args', {}).items():
                toml_lines.append(f'args.{arg_name}.doc = """{d["doc"]}"""')


    # Write

    toml_lines.append("")
    toml = "\n".join(toml_lines)

    with open(DEFINITIONS_DIR / fname, "wb") as f:
        f.write(toml.encode())

    # Check that file is correct toml and structure
    read_definition_file(fname)



if __name__ == "__main__":
    unite_toml_files()
