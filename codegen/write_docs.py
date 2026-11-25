"""Generate RST text that describes the differences between the base API
and the WebGPU API, as defined in diff_base.toml."""

from codegen_defs import read_definition_file
from codegen_utils import print, read_file, write_file


def write_docs():
    """Write the differences of the base API (relative to the WebGPU spec) to the docs."""

    diff_api = read_definition_file("diff_base.toml")
    base_api = read_definition_file("webgpu.toml")
    text = diff_api_to_rst(diff_api, base_api).rstrip()
    print(f"Writing {len(text.splitlines())} API differences to docs/wgpu.rst")

    fname = ("docs", "wgpu.rst")
    code = read_file(*fname)
    code = insert_diff_text(code, text)
    write_file(code, *fname)


def insert_diff_text(code: str, text: str) -> str:
    """Insert the given text in-between the "api-difference" markers."""
    marker1 = ".. api-difference-start"
    marker2 = ".. api-difference-end"
    if marker1 not in code or marker2 not in code:
        raise RuntimeError("Cannot find the api-difference markers in docs/wgpu.rst")
    part1, _, tail = code.partition(marker1)
    _, _, part2 = tail.partition(marker2)
    return part1 + marker1 + "\n\n" + text + "\n\n" + marker2 + part2


def diff_api_to_rst(diff_api: dict, base_api: dict) -> str:
    """Produce RST text (a bullet list) that describes the changes defined
    in the given diff, relative to the given base. Each bullet starts with
    "Removed:", "Added:", or "Changed:".
    """
    bullets = []
    for section, section_dict in diff_api.items():
        if section == "meta" or not section_dict:
            continue
        for name, item_dict in section_dict.items():
            base_item = base_api.get(section, {}).get(name)
            _collect_bullets(item_dict, base_item, (section, name), bullets)
    return "\n".join(bullets) + "\n"


def _collect_bullets(diff_node, base_node, path, bullets) -> None:
    """Walk the given diff node (relative to the corresponding base node),
    and collect a bullet for each removal, addition, or change.
    """
    if diff_node.get("diff") == "REMOVE":
        bullets.append(f"* Removed: {_describe_entity(path)}.")
        return

    if base_node is None:
        # Not present in the base API, so this is an addition in its entirety
        bullets.append(f"* Added: {_describe_entity(path)}.")
        return

    # Scalar changes. Note that docs are not part of an API diff.
    changes = {}
    for key, val in diff_node.items():
        if key in ("diff", "doc") or isinstance(val, dict):
            continue
        if not isinstance(base_node, dict) or base_node.get(key) != val:
            changes[key] = val
    if changes:
        bullets.append(
            f"* Changed: {_describe_entity(path)} {_describe_changes(changes)}."
        )

    # Recurse into sub-items (fields, methods, args, returns)
    for key, val in diff_node.items():
        if isinstance(val, dict):
            sub_base = base_node.get(key) if isinstance(base_node, dict) else None
            _collect_bullets(val, sub_base, (*path, key), bullets)


def _describe_entity(path) -> str:
    """Describe the item at the given path in words, e.g.
    "the ``GPUAdapter.summary`` property", or
    "the ``power_preference`` field of the ``RequestAdapterOptions`` struct".
    """
    names = {
        "flags": "flag",
        "enums": "enum",
        "structs": "struct",
        "classes": "class",
    }
    kind = names.get(path[0], path[0])

    if len(path) == 2:
        return f"the ``{path[1]}`` {kind}"
    elif path[2] == "fields":
        return f"the ``{path[3]}`` field of the ``{path[1]}`` {kind}"
    elif path[0] == "classes" and len(path) >= 4:
        stub = f"``{path[1]}.{path[3]}``"
        kind = "method" if path[2] == "methods" else "property"
        if len(path) == 4:
            return f"the {stub} {kind}"
        elif path[4] == "args":
            return f"the ``{path[5]}`` argument of the {stub} {kind}"
        elif path[4] == "returns":
            return f"the return value of the {stub} {kind}"
    return f"``{'.'.join(path)}``"


def _describe_changes(changes: dict) -> str:
    """Describe the given scalar changes in words, e.g.
    "is now ``int``, and defaults to ``1``"."""
    parts = []
    if "type" in changes:
        parts.append(f"is now ``{changes['type']}``")
    if "bases" in changes:
        bases = changes["bases"]
        if bases:
            bases_str = ", ".join(f"``{b}``" for b in bases)
            parts.append(f"now inherits from {bases_str}")
        else:
            parts.append("no longer has bases")
    if "default" in changes:
        default = changes["default"]
        if default:
            parts.append(f"defaults to ``{default}``")
        else:
            parts.append("no longer has a default value")
    for key in changes:
        if key not in ("type", "bases", "default"):
            parts.append(f"has a new value for {key}")  # future-proof fallback
    return ", ".join(parts)


if __name__ == "__main__":
    write_docs()
