// Serve a directory and open a page in headless Chrome with WebGPU enabled.
// Echoes the page console; exits 0 when the page prints "ALL OK", 1 on "FAIL" or timeout.
//
//   node run_chrome.mjs <dir> <page-with-query> [timeout_s]
//
// Env: CHROME_PATH (default: /usr/bin/google-chrome), CHROME_FLAGS (extra flags,
// e.g. "--use-angle=vulkan --ignore-gpu-blocklist" for a hardware GPU locally),
// SCREENSHOT (path to save a screenshot of the page at the end).
import puppeteer from 'puppeteer-core';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const [dir, page, timeout = '300'] = process.argv.slice(2);
const types = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json', '.py': 'text/plain',
  '.zip': 'application/zip', '.whl': 'application/zip',
};
const server = http.createServer((req, res) => {
  const p = path.join(dir, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  fs.readFile(p, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(0);

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--no-sandbox',
         ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean)],
});
const tab = await browser.newPage();
let done = false;
async function finish(code) {
  if (done) return;
  done = true;
  if (process.env.SCREENSHOT) await tab.screenshot({ path: process.env.SCREENSHOT }).catch(() => {});
  await browser.close();
  server.close();
  process.exit(code);
}
tab.on('console', (m) => {
  const t = m.text();
  console.log(t);
  if (t.includes('ALL OK')) finish(0);
  else if (t.startsWith('FAIL: ')) finish(1);
});
tab.on('pageerror', (e) => { console.log('pageerror: ' + e.message); finish(1); });
setTimeout(() => { console.log('FAIL: timeout'); finish(1); }, Number(timeout) * 1000);
await tab.goto(`http://localhost:${server.address().port}/${page}`);
