"""Parse the WebGPU Typescript type definitions (index.d.ts) to extract docs for most objects."""

import re

from apigen_utils import read_file, read_definition_file, DEFINITIONS_DIR, to_snake_case

KINDS = [
    "flag",
    "flagfield",
    "enum",
    "enumfield",
    "struct",
    "structfield",
    "class",
    "property",
    "method",
    "arg",
    "returns",
]
OPEN, CLOSE = "([{<", ")]}>"


def parse_ts_to_toml():
    """Parse the typescript file to a TOML structure."""
    source = read_file("apigen", "definitions", "index.d.ts")
    docs = parse(source)

    flat = {"flags": {}, "enums": {}, "structs": {}, "classes": {}}

    category_map = {
        "flag": "flags",
        "flagfield": "flags",
        "enum": "enums",
        "enumfield": "enums",
        "struct": "structs",
        "structfield": "structs",
        "class": "classes",
        "property": "classes",
        "method": "classes",
        "arg": "classes",
        "returns": "classes",
    }

    # First we flatten the docs
    for category, d in docs.items():
        for k, v in d.items():
            section = category_map[category]
            target_dict = flat[section]
            name_parts = k.split(".")
            if category in ("structfield", "property", "method", "arg", "returns"):
                name_parts[1] = to_snake_case(name_parts[1])
            if category in ("arg", ):
                name_parts[-1] = to_snake_case(name_parts[-1])
            if category.endswith("field"):
                name_parts.insert(1, "fields")
            if category == "returns":
                name_parts.append("returns")
            if category == "arg":
                name_parts.insert(-1, "args")
            if category in ("property", "method", "arg", "returns"):
                name_parts.insert(1, "methods")
            if section != "classes" and name_parts[0].startswith("GPU"):
                name_parts[0] = name_parts[0][3:]
            name_parts.append("doc")
            key = ".".join(name_parts)
            target_dict[key] = v

    # Generate TOML

    fname = "docs_ts.toml"
    hline = "# " + "-" * 80
    hline3 = "\n" + hline + "\n" + hline + "\n" + hline

    toml_lines = [
       f"# {fname}: Docs generated from 'index.d.ts'.",
    ]
    for section_name, d in flat.items():
        toml_lines.append(hline3)
        toml_lines.append(f"[{section_name}]")
        keys = sorted(d.keys())
        prev_prefix = ""
        for key in keys:
            doc = d[key]
            if doc:
                prefix = key.split(".")[0]
                if prefix != prev_prefix:
                    toml_lines.append("")
                    prev_prefix = prefix
                if "\n" not in doc:
                    toml_lines.append(f"{key} = {doc!r}")
                else:
                    toml_lines.append(f"{key} = '''\n{doc}\n'''")

    # Write

    toml_lines.append("")
    toml = "\n".join(toml_lines)

    with open(DEFINITIONS_DIR  / fname, "wb") as f:
        f.write(toml.encode())

    # Check that file is correct toml and structure
    read_definition_file(fname)


def clean_doc(raw):
    """'/** ... */' -> plain text lines."""
    body = re.sub(r"^\s*/\*\*", "", raw)
    body = re.sub(r"\*/\s*$", "", body)
    lines = [re.sub(r"^\s*\*? ?", "", l).rstrip() for l in body.split("\n")]
    return "\n".join(lines).strip()


def split_doc(doc):
    """-> (description, {param: text}, returns_text)"""
    desc, params, ret = [], {}, ""
    cur = None
    for line in doc.split("\n"):
        m = re.match(r"@param\s+(?:\{[^}]*\}\s*)?(\w+)\s*-?\s*(.*)", line)
        if m:
            cur = m.group(1)
            params[cur] = m.group(2)
            continue
        if line.startswith("@") and not line.startswith("@deprecated"):
            cur = None
            continue
        if cur and line and not line.startswith("@"):
            params[cur] += " " + line.strip()
        else:
            cur = None
            desc.append(line)
    d = "\n".join(desc).strip()
    m = re.search(r"\*\*Returns:\*\*\s*(.*)", d)
    if m:
        ret = re.sub(r"</?div[^>]*>", "", m.group(1)).strip()
    return d, params, ret


def scan_balanced(s, i):
    """s[i] is an opener; return index just past its matching closer.
    '=>' and '>' inside arrows are not treated as closers of '<'."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c in "([{<":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ">" and s[i - 1] != "=":
            depth -= 1
        if depth == 0:
            return i + 1
        i += 1
    return i


def block_end(s, i):
    """s[i] == '{'; return index past the matching '}', skipping comments/strings."""
    depth, n = 0, len(s)
    while i < n:
        if s.startswith("/*", i):
            i = s.index("*/", i) + 2
            continue
        if s.startswith("//", i):
            nl = s.find("\n", i)
            i = n if nl < 0 else nl
            continue
        c = s[i]
        if c in "\"'`":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def split_top(s, sep=","):
    out, depth, cur = [], 0, []
    for i, c in enumerate(s):
        if c in "([{<":
            depth += 1
        elif c in ")]}" or (c == ">" and s[i - 1] != "="):
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def tokenize_members(body):
    """Yield (doc_or_None, member_text) from an interface body."""
    i, n = 0, len(body)
    doc = None
    while i < n:
        if body[i].isspace():
            i += 1
            continue
        if body.startswith("/**", i):
            j = body.index("*/", i) + 2
            doc = clean_doc(body[i:j])
            i = j
            continue
        if body.startswith("/*", i):
            i = body.index("*/", i) + 2
            continue
        if body.startswith("//", i):
            nl = body.find("\n", i)
            i = n if nl < 0 else nl
            continue
        # member: runs to ';' at depth 0
        j, depth = i, 0
        while j < n:
            c = body[j]
            if c in "([{<":
                depth += 1
            elif c in ")]}" or (c == ">" and body[j - 1] != "="):
                depth -= 1
            elif c == ";" and depth == 0:
                break
            j += 1
        yield doc, re.sub(r"\s+", " ", body[i:j]).strip()
        doc = None
        i = j + 1


def parse_member(text):
    """-> dict(name, is_method, params[list of names], ret) or None"""
    m = re.match(r"(?:readonly\s+)?(?:\[(\w+)[^\]]*\]|(\w+))\s*(\?)?\s*", text)
    if not m:
        return None
    name = m.group(2) or m.group(1)
    rest = text[m.end() :]
    if rest.startswith("<"):  # generics on a method
        rest = rest[scan_balanced(rest, 0) :].lstrip()
    if rest.startswith("("):
        end = scan_balanced(rest, 0)
        plist = rest[1 : end - 1]
        params = []
        for p in split_top(plist):
            pm = re.match(r"\s*(?:\.\.\.)?(\w+)", p)
            if pm:
                params.append(pm.group(1))
        ret = rest[end:].lstrip(": ").strip()
        return dict(name=name, method=True, params=params, ret=ret)
    ty = rest.lstrip(": ").strip()
    return dict(name=name, method=False, ty=ty)


def parse(src):
    out = {k: {} for k in KINDS}

    def put(kind, key, val):
        if val or key not in out[kind]:
            out[kind][key] = val if val else out[kind].get(key, "")

    # statements at top level
    pos, n = 0, len(src)
    pending_doc = None
    stmt_re = re.compile(r"^(?:declare\s+)?(interface|type)\s+(\w+)", re.M)
    while True:
        m = stmt_re.search(src, pos)
        if not m:
            break
        # doc comment immediately before?
        before = src[: m.start()].rstrip()
        pending_doc = None
        if before.endswith("*/"):
            st = before.rfind("/**")
            if st >= 0 and "*/" not in before[st:-2]:
                pending_doc = clean_doc(before[st:])
        kind, name = m.group(1), m.group(2)
        if kind == "type":
            end = src.index(";", m.end())
            while src[end - 1] == "\\":
                end = src.index(";", end + 1)
            body = src[m.end() : end]
            lits = re.findall(r'^\s*\|\s*"([^"]*)"', body, re.M)
            if lits and re.fullmatch(r'\s*=\s*(\|\s*"[^"]*"\s*)+', body):
                put("enum", name, split_doc(pending_doc)[0] if pending_doc else "")
                for v in lits:
                    put("enumfield", f"{name}.{v}", "")
            pos = end + 1
            continue
        # interface
        ob = src.index("{", m.end())
        cb = block_end(src, ob)
        body = src[ob + 1 : cb - 1]
        pos = cb
        members = [(d, parse_member(t)) for d, t in tokenize_members(body)]
        members = [(d, p) for d, p in members if p]
        idoc = split_doc(pending_doc)[0] if pending_doc else ""
        if not members:
            continue
        if all((not p["method"]) and p["ty"] == "GPUFlagsConstant" for _, p in members):
            put("flag", name, idoc)
            for d, p in members:
                put("flagfield", f"{name}.{p['name']}", split_doc(d)[0] if d else "")
        elif any(p["name"] == "__brand" or p["method"] for _, p in members):
            put("class", name, idoc)
            for d, p in members:
                if p["name"] == "__brand":
                    continue
                if p["method"]:
                    desc, pdocs, ret = split_doc(d) if d else ("", {}, "")
                    key = f"{name}.{p['name']}"
                    put("method", key, desc)
                    for a in p["params"]:
                        put("arg", f"{key}.{a}", pdocs.get(a, ""))
                    put("returns", key, ret)
                else:
                    put("property", f"{name}.{p['name']}", split_doc(d)[0] if d else "")
        else:
            put("struct", name, idoc)
            for d, p in members:
                put("structfield", f"{name}.{p['name']}", split_doc(d)[0] if d else "")
    return out


if __name__ == "__main__":
    parse_ts_to_toml()
