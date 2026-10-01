// Run Python in Pyodide under Node.js, with navigator.gpu provided by the
// `webgpu` npm package (Dawn's Node.js bindings).
//
//   node run_node.mjs <wheel,wheel,...> script.py
//   node run_node.mjs <wheel,wheel,...> --pytest <tests-dir> [pytest args...]
//
// Exits with a non-zero code if the script raises or pytest fails.
import { loadPyodide } from 'pyodide';
import { create, globals } from 'webgpu';
import fs from 'node:fs';
import path from 'node:path';

Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, 'gpu', { value: create([]), configurable: true });

const [wheels, ...rest] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage(rest[0] === '--pytest' ? ['cffi', 'numpy', 'pytest'] : ['cffi']);
const sitePackages = py.runPython('import site; site.getsitepackages()[0]');
for (const w of wheels.split(',').filter(Boolean)) {
  py.unpackArchive(new Uint8Array(fs.readFileSync(w)), 'wheel', { extractDir: sitePackages });
}

let exitCode = 0;
try {
  if (rest[0] === '--pytest') {
    const [, testsDir, ...args] = rest;
    py.FS.mkdirTree('/work/tests');
    py.FS.mount(py.FS.filesystems.NODEFS, { root: path.resolve(testsDir) }, '/work/tests');
    py.globals.set('pytest_args', py.toPy(args));
    // runPythonAsync runs with JSPI, so the sync wgpu API works (via pyodide.ffi.run_sync)
    exitCode = Number(await py.runPythonAsync(`
import os, sys, pytest
os.chdir("/work")
sys.path.insert(0, "/work/tests")
int(pytest.main(["-v", "-p", "no:cacheprovider", *pytest_args]))
`));
  } else {
    const script = rest[0];
    await py.runPythonAsync(fs.readFileSync(script, 'utf8') +
      '\nif "main" in globals():\n    await main()\n');
  }
} catch (e) {
  console.log(String(e));
  exitCode = 1;
}
process.exit(exitCode);
