"""
A little script that serves browser-based example, using a wheel from the local wgpu.

* Examples that run wgpu fully in the browser in Pyodide / PyScript.

What this script does:

* runs the codegen for js_webgpu backend
* Build the .whl for wgpu, so Pyodide can install the dev version.
* Start a tiny webserver to host html files for a selection of examples.
* Opens a webpage in the default browser.

Files are loaded from disk on each request, so you can leave the server running
and just update examples, update wgpu and build the wheel, etc.
"""

# run_example = false

# this is adapted from the rendercanvas version

import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess

import wgpu
from codegen import update_js, file_cache

# from here: https://github.com/harfbuzz/uharfbuzz/pull/275
uharfbuzz_wheel = "uharfbuzz-0.1.dev1+ga19185453-cp310-abi3-pyodide_2025_0_wasm32.whl"
pygfx_wheel = "pygfx-0.15.3-py3-none-any.whl"
pygfx_deps = [uharfbuzz_wheel, "hsluv", "pylinalg", "jinja2", pygfx_wheel]
rendercanvas_deps = ["sniffio", "rendercanvas==2.4.2"] # later rendercanvas or not having sniffio casues a hang in run_sync: "https://github.com/pyodide/pyodide/issues/6077

#examples that don't require a canvas, we will capture the output to a div
compute_examples = {
    # "compute_int64.py", # this one requires native only features, so won't work in the browser for now
    "compute_noop.py": [], # no deps
    "compute_matmul.py": ["numpy"],
    "compute_textures.py": ["numpy", "imageio"], #imageio doesn't work in pyodide right now (fetch?)
    "compute_timestamps.py": [], # this one still crashes as the descriptor doesn't get converted into an object...
    # try these two up here too to make sure it's not the canvas. (having them twice is a problem because only one .html variant exsits)
    "pygfx_example.py": [*pygfx_deps, "sniffio", "imageio", "imgui-bundle"], # currnetly hangs... so I digging down on that
    "fpl_example.py": [*pygfx_deps, "fastplotlib"], #also hangs but I eventually want to get to this.
}

# these need rendercanvas too for now. but might run with just a canvas (no events) in the near future.
graphics_examples = {
    "triangle.py":[], # no deps
    "cube.py": ["numpy"],
    "offscreen_hdr.py": ["numpy", "pypng"], # pyscript says it doesn't work in pyodide.
    # "triangle_glsl.py": # we can't use GLSL in the browser... I am looking into maybe using wasm compiled naga manually - at a later date.
    "imgui_backend_sea.py": ["numpy", "imgui-bundle"],
    "imgui_basic_example.py": ["imgui-bundle"], # might even work without wgpu as imgui already works in pyodide...
    "imgui_renderer_sea.py": ["numpy", "imgui-bundle"],
    # pygfx example
    # "pygfx_example.py": [*pygfx_deps, "sniffio", "imageio"],
    # "fpl_example.py": [*pygfx_deps, "fastplotlib"],
    # theoretically WGSL shadertoys work (with a couple tweaks... needs a new release soonish)
    "shadertoy_blink.py": ["numpy", "sniffio", "requests", "wgpu_shadertoy-0.2.0-py3-none-any.whl"],
}


root = os.path.abspath(os.path.join(__file__, "..", ".."))

short_version = ".".join(str(i) for i in wgpu.version_info[:3])
wheel_name = f"wgpu-{short_version}-py3-none-any.whl"


def get_html_index():
    """Create a landing page."""

    compute_examples_list = [f"<li><a href='{name.replace('.py', '.html')}'>{name}</a></li>" for name in compute_examples.keys()]
    graphics_examples_list = [f"<li><a href='{name.replace('.py', '.html')}'>{name}</a></li>" for name in graphics_examples.keys()]

    html = """<!doctype html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width,initial-scale=1.0">
        <title>wgpu PyScript examples</title>
        <script type="module" src="https://pyscript.net/releases/2025.11.2/core.js"></script>
    </head>
    <body>

    <a href='/build'>Rebuild the wheel</a><br><br>
    """

    html += "List of compute examples that run in Pyodide:\n"
    html += f"<ul>{''.join(compute_examples_list)}</ul><br>\n\n"

    html += "List of graphics examples that run in PyScript:\n"
    html += f"<ul>{''.join(graphics_examples_list)}</ul><br>\n\n"

    html += "</body>\n</html>\n"
    return html


html_index = get_html_index()


# An html template to show examples using pyscript.
pyscript_graphics_template = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>{example_script} via PyScript</title>
    <script type="module" src="https://pyscript.net/releases/2025.11.2/core.js"></script>
</head>

<body>
    <a href="/">Back to list</a><br><br>

    <p>
    {docstring}
    </p>
    <dialog id="loading" style='outline: none; border: none; background: transparent;'>
        <h1>Loading...</h1>
    </dialog>
    <script type="module">
        const loading = document.getElementById('loading');
        addEventListener('py:ready', () => loading.close());
        loading.showModal();
    </script>

    <canvas id="canvas" style="background:#aaa; width: 90%; height: 480px;"></canvas>
    <script type="py" src="{example_script}",
        config='{{"packages": [{dependencies}]}}'>
    </script>
</body>

</html>
"""

# TODO: a pyodide example for the compute examples (so we can capture output?)
# modified from _pyodide_iframe.html from rendercanvas
pyodide_compute_template = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>{example_script} via Pyodide</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js"></script>
</head>

<dialog id="loading" style='outline: none; border: none; background: transparent;'>
        <h1>Loading...</h1>
    </dialog>
<body>
    <a href="/">Back to list</a><br><br>
    <p>
    {docstring}
    </p>
    <div id="output" style="white-space: pre-wrap; background:#eee; padding:4px; margin:4px; border:1px solid #ccc;">
        <p>Output:</p>
    </div>
    <canvas id='canvas' style='width:calc(100% - 20px); height: 450px; background-color: #ddd;'></canvas>
    <script type="text/javascript">
        async function main() {{
            let loading = document.getElementById('loading');
            loading.showModal();
            try {{
                let example_name = {example_script!r};
                pythonCode = await (await fetch(example_name)).text();
                // this env var is really only used for the pygfx examples - so maybe we make a script for that gallery instead?
                let pyodide = await loadPyodide({{env: {{'PYGFX_DEFAULT_PPAA': 'none' }}}});
                pyodide.setStdout({{
                    batched: (s) => {{
                        // TODO: newline, scrollable, echo to console?
                        document.getElementById("output").innerHTML += "<br>" + s.replace(/</g, "&lt;").replace(/>/g, "&gt;");
                        console.log(s); // so we also have it formatted
                    }}
                }});
                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                // TODO: maybe use https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.loadPackagesFromImports
                {dependencies}
                pyodide.setDebug(true);
                let ret = await pyodide.runPythonAsync(pythonCode);
                console.log("Example finished:", ret);
                loading.close();
            }} catch (err) {{
                // TODO: this could be formatted better as this overlaps and is unreadable...
                loading.innerHTML = "Failed to load: " + err;
                console.error(err); // so we have it here too
            }}
        }}
        main();
    </script>
</body>

</html>
"""




if not (
    os.path.isfile(os.path.join(root, "wgpu", "__init__.py"))
    and os.path.isfile(os.path.join(root, "pyproject.toml"))
):
    raise RuntimeError("This script must run in a checkout repo of wgpu-py.")


def build_wheel():
    # TODO: run the codegen for js_webgpu backend!
    file_cache.reset()
    update_js()
    file_cache.write_changed_files_to_disk()
    # doesn't work right now :/
    # maybe now? not 100% sure

    # TODO: can we use the existing hatch build system? (via build yes, not via flit it seems)
    os.environ["WGPU_BUILD_PLATFORM_INFO"] = " ".join(("pyodide_wasm", "any"))
    subprocess.run([sys.executable, "-m", "build", "-n", "-w"], cwd=root)
    wheel_filename = os.path.join(root, "dist", wheel_name)
    assert os.path.isfile(wheel_filename), f"{wheel_name} does not exist"


def get_docstring_from_py_file(fname):
    filename = os.path.join(root, "examples", fname)
    docstate = 0
    doc = ""
    with open(filename, "rb") as f:
        for line in f:
            line = line.decode()
            if docstate == 0:
                if line.lstrip().startswith('"""'):
                    docstate = 1
            else:
                if docstate == 1 and line.lstrip().startswith(("---", "===")):
                    docstate = 2
                    doc = ""
                elif '"""' in line:
                    doc += line.partition('"""')[0]
                    break
                else:
                    doc += line

    return doc.replace("\n\n", "<br><br>")


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.respond(200, html_index, "text/html")
        elif self.path == "/build":
            # TODO: add progress instead of blocking before load?
            # also seems like this might get called multiple times?
            try:
                # this doesn't seem to work correctly. works on startup but not upon visiting this page.
                # I think it doesn't see updates to jswriter.py while the server is running.
                build_wheel()
            except Exception as err:
                self.respond(500, str(err), "text/plain")
            else:
                html = f"Wheel build: {wheel_name}<br><br><a href='/'>Back to list</a>"
                self.respond(200, html, "text/html")
        elif self.path.endswith(".whl"):
            filename = os.path.join(root, "dist", self.path.strip("/"))
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                self.respond(200, data, "application/octet-stream")
            else:
                self.respond(404, "wheel not found")
        elif self.path.endswith(".html"):
            name = self.path.strip("/")
            pyname = name.replace(".html", ".py")
            if pyname in graphics_examples:
                deps = graphics_examples[pyname].copy() # don't modify them multiple times!
                deps = [*rendercanvas_deps, *deps] # so the rendercanvas deps aren't overwritten by rendercanvas being a dependency of pygfx... surely there is a better way.
                deps.append(f"./{wheel_name}")
                # sometimes sniffio is missing, other times it's not?
                doc = get_docstring_from_py_file(pyname)
                html = pyscript_graphics_template.format(docstring=doc, example_script=pyname, dependencies=", ".join([f'"{d}"' for d in deps]))
                self.respond(200, html, "text/html")
            elif pyname in compute_examples:
                doc = get_docstring_from_py_file(pyname)
                deps = compute_examples[pyname].copy()
                # deps.append(f"./{wheel_name}") # putting this at the end might not always be the correct way...
                deps = [f"/{wheel_name}", *deps]
                deps = [*rendercanvas_deps, *deps] # there actually is a better way for pyodide. you can put multiple deps in the install call: https://micropip.pyodide.org/en/stable/project/api.html#micropip.install
                html = pyodide_compute_template.format(docstring=doc, example_script=pyname, dependencies="\n".join([f"await micropip.install({dep!r});" for dep in deps]))
                self.respond(200, html, "text/html")
            else:
                self.respond(404, "example not found")
        elif self.path.endswith(".py"):
            filename = os.path.join(root, "examples", self.path.strip("/"))
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                self.respond(200, data, "text/plain")
            else:
                self.respond(404, "py file not found")
        else:
            self.respond(404, "not found")

    def respond(self, code, body, content_type="text/plain"):
        self.send_response(code)
        self.send_header("Content-type", content_type)
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[-1])
        except ValueError:
            pass

    build_wheel()
    print("Opening page in web browser ...")
    webbrowser.open(f"http://localhost:{port}/")
    HTTPServer(("", port), MyHandler).serve_forever()
