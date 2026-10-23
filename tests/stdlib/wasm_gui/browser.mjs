// std/gui on wasm32 in a real browser (headless Chromium through
// Playwright): examples/gui_gallery built for wasm32 runs in a page through
// std/gui/gui_web.js, with the DejaVu fonts the native build found served
// through the glue's virtual files; its widgets window is compared pixel by
// pixel with the native headless snapshot of the same frame
// (gui_gallery --snapshot), then driven with real mouse and keyboard input.
// Also loads examples/gui_web at devicePixelRatio 2 and checks it animates.
//   usage: node browser.mjs <gallery.wasm> <snapshot dir> <gui_web.wasm> <gui_web.js> <out dir> <examples/gui_web/web/index.html> <bundle dir> <font>...
//   Env: NODE_PATH must find 'playwright'; PLAYWRIGHT_BROWSERS_PATH its browsers.
import { createServer } from "node:http";
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { basename } from "node:path";
import { createRequire } from "node:module";

const [galleryWasm, snapDir, webWasm, glue, outDir, examplePage, bundleDir, ...fonts] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
mkdirSync(outDir, { recursive: true });

const fontMap = Object.fromEntries(fonts.map((f) => [f, `/fonts/${basename(f)}`]));
const page = (wasm, extra = {}) => `<!doctype html><html><head><meta charset="utf-8"><title>t</title>
<style>body{margin:0;background:#888} canvas{display:block;margin:0 0 8px 0}</style></head><body>
<script type="module">
import { loadSafeCGui, fetchFiles } from "/gui_web.js";
const files = await fetchFiles(${JSON.stringify(fontMap)});
window.app = await loadSafeCGui(fetch("${wasm}"), { files, title: true, ...${JSON.stringify(extra)} });
window.ready = true;
</script></body></html>`;
const routes = {
    "/gallery.html": ["text/html", page("/gallery.wasm")],
    "/web.html": ["text/html", page("/web.wasm")],
    "/gallery-webgl.html": ["text/html", page("/gallery.wasm", { present: "webgl", preserveDrawingBuffer: true })],
    "/web-frames.html": ["text/html", page("/web.wasm", { windows: "frames" })],
    "/gui_web.js": ["text/javascript", readFileSync(glue)],
    "/gallery.wasm": ["application/wasm", readFileSync(galleryWasm)],
    "/web.wasm": ["application/wasm", readFileSync(webWasm)],
    "/snap/widgets.png": ["image/png", readFileSync(`${snapDir}/widgets.png`)],
};
for (const f of fonts) routes[`/fonts/${basename(f)}`] = ["font/ttf", readFileSync(f)];
// examples/gui_web/web as its header says to serve it
routes["/example/index.html"] = ["text/html", readFileSync(examplePage)];
routes["/example/gui_web.wasm"] = routes["/web.wasm"];
routes["/example/gui_web.js"] = routes["/gui_web.js"];
for (const f of fonts) routes[`/example/fonts/${basename(f)}`] = routes[`/fonts/${basename(f)}`];
// 'safeguard bundle --target wasm32' output, as is
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".png": "image/png" };
for (const f of readdirSync(bundleDir)) routes[`/bundle/${f}`] = [types[f.slice(f.lastIndexOf("."))] || "application/octet-stream", readFileSync(`${bundleDir}/${f}`)];
const server = createServer((req, res) => {
    const r = routes[req.url];
    if (!r) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": r[0] });
    res.end(r[1]);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

let failures = 0;
const expect = (name, cond, extra = "") => {
    console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}${cond ? "" : " " + extra}`);
    if (!cond) failures++;
};

// WebGL without a GPU: SwiftShader through ANGLE
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
    // ── the gallery at devicePixelRatio 1, against the native frame ──
    const ctx = await browser.newContext({ viewport: { width: 900, height: 1300 }, deviceScaleFactor: 1 });
    const p = await ctx.newPage();
    const errors = [];
    p.on("pageerror", (e) => errors.push(String(e)));
    p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    await p.goto(`${base}/gallery.html`);
    await p.waitForFunction(() => window.ready === true, null, { timeout: 30000 }).catch((e) => { throw new Error(`gallery page not ready: ${errors.join("; ")}`); });
    await p.waitForTimeout(300);
    expect("gallery loaded, no page errors", errors.length === 0, errors.join("; "));
    const info = await p.evaluate(() => {
        const c = document.querySelectorAll("canvas");
        return { n: c.length, w: c[0].width, h: c[0].height, title: document.title, running: window.app.gui.running };
    });
    expect("two canvases, 860x620 widgets window, frame loop running", info.n === 2 && info.w === 860 && info.h === 620 && info.running, JSON.stringify(info));
    expect("document.title from the first window", info.title.startsWith("SafeC GUI gallery"));
    const cmp = await p.evaluate(async () => {
        const img = new Image();
        img.src = "/snap/widgets.png";
        await img.decode();
        const ref = document.createElement("canvas");
        ref.width = img.width; ref.height = img.height;
        const rc = ref.getContext("2d");
        rc.drawImage(img, 0, 0);
        const a = rc.getImageData(0, 0, img.width, img.height).data;
        const live = document.querySelectorAll("canvas")[0];
        const b = live.getContext("2d").getImageData(0, 0, live.width, live.height).data;
        if (a.length !== b.length) return { sizeMismatch: [img.width, img.height, live.width, live.height] };
        let exact = 0, close = 0, maxd = 0;
        const n = a.length / 4;
        for (let i = 0; i < a.length; i += 4) {
            const d = Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]), Math.abs(a[i + 3] - b[i + 3]));
            if (d === 0) exact++;
            if (d <= 2) close++;
            if (d > maxd) maxd = d;
        }
        return { n, exact, close, maxd };
    });
    console.log(`    widgets window vs native snapshot: ${cmp.exact}/${cmp.n} pixels identical, ${cmp.close} within 2, max difference ${cmp.maxd}`);
    // The only pixels that may differ are the two time-driven bars (the
    // indeterminate progress bar and the busy indicator, ~700 pixels): the
    // browser's clock isn't the headless backend's virtual one.
    expect("the browser frame matches the native one (all but the animated bars)", !cmp.sizeMismatch && cmp.n - cmp.close <= 1500, JSON.stringify(cmp));
    await p.locator("canvas").first().screenshot({ path: `${outDir}/gallery-widgets.png` });
    await p.locator("canvas").nth(1).screenshot({ path: `${outDir}/gallery-canvas.png` });
    // the canvas window animates
    const sum = () => p.evaluate(() => {
        const c = document.querySelectorAll("canvas")[1];
        const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
        let h = 0;
        for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) >>> 0;
        return h;
    });
    const s1 = await sum();
    await p.waitForTimeout(200);
    expect("canvas window animates (requestAnimationFrame + on_frame)", (await sum()) !== s1);
    // real input: click the second tab, then the first text field region
    const before = await p.evaluate(() => document.querySelectorAll("canvas")[0].toDataURL());
    const box = await p.locator("canvas").first().boundingBox();
    await p.mouse.move(box.x + 400, box.y + 300);
    await p.mouse.wheel(0, 200);
    for (const x of [130, 200, 270]) {   // the tab strip under the menu bar
        await p.mouse.click(box.x + x, box.y + 42);
        await p.waitForTimeout(60);
    }
    const after = await p.evaluate(() => document.querySelectorAll("canvas")[0].toDataURL());
    expect("mouse clicks change the widgets window (tabs)", before !== after);
    const focused = await p.evaluate(() => document.activeElement && document.activeElement.tagName);
    expect("a click gives the window's hidden input the DOM focus", focused === "TEXTAREA", String(focused));
    await p.keyboard.type("abc");
    await p.keyboard.press("Tab");
    await p.keyboard.insertText("日本語");   // text with no key events (IME commit / virtual keyboard)
    await p.waitForTimeout(100);
    expect("keyboard input handled without page errors", errors.length === 0, errors.join("; "));
    await p.locator("canvas").first().screenshot({ path: `${outDir}/gallery-widgets-after-input.png` });
    await ctx.close();

    // ── the gallery through WebGL, against the putImageData frame ──
    const ctxG = await browser.newContext({ viewport: { width: 900, height: 1300 }, deviceScaleFactor: 1 });
    const pg = await ctxG.newPage();
    const errsG = [];
    pg.on("pageerror", (e) => errsG.push(String(e)));
    await pg.goto(`${base}/gallery-webgl.html`);
    await pg.waitForFunction(() => window.ready === true, null, { timeout: 30000 });
    await pg.waitForTimeout(300);
    const kind = await pg.evaluate(() => window.app.gui.presenter(window.app.gui.windows()[0].id));
    expect("WebGL: the gallery presents through WebGL2", kind === "webgl2", String(kind));
    const cmpG = await pg.evaluate(async () => {
        const img = new Image();
        img.src = "/snap/widgets.png";
        await img.decode();
        const grab = (src) => {
            const c = document.createElement("canvas");
            c.width = 860; c.height = 620;
            const cx = c.getContext("2d");
            cx.drawImage(src, 0, 0);
            return cx.getImageData(0, 0, 860, 620).data;
        };
        const a = grab(img), b = grab(document.querySelectorAll("canvas")[0]);
        let close = 0;
        for (let i = 0; i < a.length; i += 4) {
            const d = Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]));
            if (d <= 2) close++;
        }
        return { n: a.length / 4, close };
    });
    console.log(`    WebGL widgets window vs native snapshot: ${cmpG.close}/${cmpG.n} pixels within 2`);
    expect("WebGL: the same frame as the CPU path (all but the animated bars)", cmpG.n - cmpG.close <= 1500, JSON.stringify(cmpG));
    await pg.locator("canvas").first().screenshot({ path: `${outDir}/gallery-widgets-webgl.png` });
    const sumG = () => pg.evaluate(() => { const c = document.createElement("canvas"); c.width = 470; c.height = 560; const x = c.getContext("2d"); x.drawImage(document.querySelectorAll("canvas")[1], 0, 0); const d = x.getImageData(0, 0, 470, 560).data; let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) >>> 0; return h; });
    const g1 = await sumG();
    await pg.waitForTimeout(200);
    expect("WebGL: the canvas window animates", (await sumG()) !== g1);
    expect("WebGL: no page errors", errsG.length === 0, errsG.join("; "));
    await ctxG.close();

    // ── examples/gui_web in frames ──
    const ctxF = await browser.newContext({ viewport: { width: 900, height: 700 }, deviceScaleFactor: 1 });
    const pf = await ctxF.newPage();
    await pf.goto(`${base}/web-frames.html`);
    await pf.waitForFunction(() => window.ready === true, null, { timeout: 30000 });
    await pf.waitForTimeout(200);
    const fr = await pf.evaluate(() => Array.from(document.querySelectorAll(".safec-gui-window")).map((f) => [f.offsetLeft, f.offsetTop]));
    const bar = pf.locator(".safec-gui-titlebar").first();
    const bb = await bar.boundingBox();
    await pf.mouse.move(bb.x + 60, bb.y + 10);
    await pf.mouse.down();
    await pf.mouse.move(bb.x + 260, bb.y + 110, { steps: 5 });
    await pf.mouse.up();
    const moved = await pf.evaluate(() => document.querySelector(".safec-gui-window").offsetLeft);
    expect("frames: two windows, dragged by the title bar", fr.length === 2 && moved === fr[0][0] + 200, JSON.stringify([fr, moved]));
    await pf.screenshot({ path: `${outDir}/gui_web-frames.png` });
    await ctxF.close();

    // ── examples/gui_web at devicePixelRatio 2 ──
    const ctx2 = await browser.newContext({ viewport: { width: 800, height: 900 }, deviceScaleFactor: 2 });
    const q = await ctx2.newPage();
    const errs2 = [];
    q.on("pageerror", (e) => errs2.push(String(e)));
    await q.goto(`${base}/web.html`);
    await q.waitForFunction(() => window.ready === true, null, { timeout: 30000 });
    await q.waitForTimeout(300);
    const i2 = await q.evaluate(() => {
        const c = document.querySelectorAll("canvas");
        return { n: c.length, w: c[0].width, cssw: c[0].style.width, w2: c[1].width };
    });
    expect("gui_web: hiDpi canvases (backing store 2x the CSS size)", i2.n === 2 && i2.w === 720 && i2.cssw === "360px" && i2.w2 === 720, JSON.stringify(i2));
    const b2 = await q.locator("canvas").first().boundingBox();
    const shot1 = await q.evaluate(() => document.querySelectorAll("canvas")[0].toDataURL());
    await q.mouse.click(b2.x + 60, b2.y + 70);   // "Click me"
    await q.waitForTimeout(100);
    const shot2 = await q.evaluate(() => document.querySelectorAll("canvas")[0].toDataURL());
    expect("gui_web: the button click redraws the window", shot1 !== shot2);
    expect("gui_web: no page errors", errs2.length === 0, errs2.join("; "));
    await q.screenshot({ path: `${outDir}/gui_web.png` });
    // the example's own page
    await q.goto(`${base}/example/index.html`);
    await q.waitForTimeout(800);
    const i3 = await q.evaluate(() => Array.from(document.querySelectorAll("canvas")).map((c) => c.width));
    expect("examples/gui_web/web/index.html runs", i3.length === 2 && i3[0] === 720 && errs2.length === 0, JSON.stringify(i3) + errs2.join("; "));
    await q.screenshot({ path: `${outDir}/gui_web-page.png` });
    // the page 'safeguard bundle --target wasm32' generated
    await q.goto(`${base}/bundle/index.html`);
    await q.waitForTimeout(800);
    const i4 = await q.evaluate(() => ({ n: document.querySelectorAll("#safec-app canvas").length, err: document.getElementById("safec-error").textContent, title: document.title }));
    expect("safeguard bundle --target wasm32: the generated page runs", i4.n === 2 && i4.err === "" && i4.title === "Controls" && errs2.length === 0, JSON.stringify(i4));
    await q.screenshot({ path: `${outDir}/gui_web-bundle.png` });
    await ctx2.close();
} finally {
    await browser.close();
    server.close();
}
console.log(failures === 0 ? "wasm gui (Chromium): all passed" : `wasm gui (Chromium): ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
