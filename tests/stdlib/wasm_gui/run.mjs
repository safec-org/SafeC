// Drives the wasm_gui test module (build/wasm_gui_tests, built by run.sh)
// through the real std/gui/gui_web.js against fake_canvas.mjs.
//   usage: node run.mjs <module.wasm> <path/to/gui_web.js> [<path/to/safec_dom.js>]
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { createFakeCanvasWindow } from "./fake_canvas.mjs";
import { Event } from "../wasm/fake_dom.mjs";

const [wasmPath, gluePath, domPath] = process.argv.slice(2);
const { loadSafeCGui } = await import(pathToFileURL(gluePath).href);
const window = createFakeCanvasWindow();
const { document } = window;
const logs = [];
const fakeConsole = { log: (m) => logs.push(m), error: (m) => logs.push(m) };
const fitCanvas = window.makeCanvas({ "data-safec-gui": "", "data-safec-fit": "" });
fitCanvas.cssBox = [300, 200];
const globals = { ...window, window, console: fakeConsole };
// the canvas option: the first two windows get new canvases, the third the page's fit canvas
let made = 0;
const canvasFor = (index, title) => (index < 2 ? window.makeCanvas({ "data-title": title }) : fitCanvas);
const opts = { globals, canvas: canvasFor, title: true, entry: "start" };
if (domPath) opts.loadSafeC = (await import(pathToFileURL(domPath).href)).loadSafeC;
const app = await loadSafeCGui(readFileSync(wasmPath), opts);
const ex = app.exports;
let failures = 0;
const expect = (name, cond, extra = "") => {
    console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}${cond ? "" : " " + extra}`);
    if (!cond) failures++;
};
const frame = (n = 1) => { for (let i = 0; i < n; i++) { window.advance(17); window.runFrame(window.performance.now()); } };
const canvases = Array.from(document.querySelectorAll("canvas"));
const cA = app.gui.canvas(ex.win_id(0)), cB = app.gui.canvas(ex.win_id(1)), cC = app.gui.canvas(ex.win_id(2));
const str = (s) => {
    const b = new TextEncoder().encode(s);
    const p = ex.scratch();
    new Uint8Array(ex.memory.buffer, p, b.length + 1).set([...b, 0]);
};
// An event at CSS position (x, y) of a canvas (its rect starts at 10, 20).
const at = (c, type, x, y, init = {}) => c.dispatchEvent(new Event(type, { clientX: 10 + x, clientY: 20 + y, ...init }));

expect("three windows, three canvases", cA && cB && cC && cA !== cB && cC === fitCanvas && canvases.length === 3);
expect("first frame ran (start + rAF loop)", window.hasFrame() && ex.frames() >= 1);
expect("CSS size = logical size", cA.style.width === "200px" && cA.style.height === "150px");
expect("backing store = logical * devicePixelRatio (hiDpi)", cA.width === 400 && cA.height === 300 && ex.win_w(0) === 400 && ex.win_scale1000(0) === 2000);
expect("hiDpi off: scale 1", ex.win_scale1000(2) === 1000);
expect("fit canvas: the page's size", ex.win_lw(2) === 300 && ex.win_w(2) === 300 && cC.width === 300 && cC.height === 200);
expect("title: aria-label and document.title", cA.getAttribute("aria-label") === "Widgets" && document.title === "Widgets");
expect("canvas focusable", cA.getAttribute("tabindex") === "0");
frame();
expect("presented: window B red", JSON.stringify(cB.readPixel(2, 2)) === "[255,0,0,255]", JSON.stringify(cB.readPixel(2, 2)));
expect("premultiplied -> straight alpha", JSON.stringify(cB.readPixel(11, 11)) === "[255,255,255,128]", JSON.stringify(cB.readPixel(11, 11)));
expect("presented: window A painted (widgets)", cA.readPixel(1, 1)[3] === 255);
const putsBefore = cB.getContext("2d").puts;
frame(3);
expect("animation: B redrawn each frame (on_frame)", cB.getContext("2d").puts > putsBefore && ex.frames() >= 4);
const putsA = cA.getContext("2d").puts;
frame(2);
expect("unchanged window not re-presented", cA.getContext("2d").puts === putsA);

// pointer
at(cB, "pointermove", 30, 20);
frame();
expect("pointer move in buffer pixels (x2)", ex.last_x() === 60 && ex.last_y() === 40, `${ex.last_x()},${ex.last_y()}`);
at(cB, "pointerdown", 15, 10, { button: 2, pointerId: 1 });
frame();
expect("right button down", ex.kind_count(3) === 1 && ex.last_button() === 1 && cB.captured === 1);
at(cB, "pointerup", 15, 10, { button: 2 });
at(cB, "wheel", 5, 5, { deltaY: 100, deltaMode: 0 });
at(cB, "wheel", 5, 5, { deltaY: -3, deltaMode: 1 });
frame();
expect("wheel: pixels and lines to lines (+y = up)", ex.kind_count(9) === 2 && ex.last_dy() === 1, String(ex.last_dy()));
expect("focus on click", document.activeElement === cB && ex.win_focused(1) === 1 && ex.kind_count(10) === 1);
// keys
const key = (c, init) => {
    const d = new Event("keydown", init);
    c.dispatchEvent(d);
    c.dispatchEvent(new Event("keyup", init));
    return d;
};
let d = key(cB, { key: "a", code: "KeyA" });
frame();
expect("key 'a': keycode A, codepoint a, default prevented", ex.last_key() === 65 && ex.last_cp() === 97 && d.defaultPrevented);
key(cB, { key: "Ä", code: "Quote", shiftKey: true });
frame();
expect("non-ASCII text, Shift", ex.last_cp() === 0xC4 && ex.last_mods() === 1);
key(cB, { key: "ArrowLeft", code: "ArrowLeft", altKey: true });
frame();
expect("arrow key + Alt, no text", ex.last_key() === 1 && ex.last_cp() === 0 && ex.last_mods() === 4);
key(cB, { key: "s", code: "KeyS", ctrlKey: true });
frame();
expect("Ctrl+S: no text", ex.last_key() === 83 && ex.last_cp() === 0 && ex.last_mods() === 2);
key(cB, { key: "F5", code: "F5" });
frame();
expect("F5", ex.last_key() === 20);
expect("primary modifier is Ctrl here", ex.primary_mod() === 2);
// widgets: click the button, type into the text field
const bx = ex.button_x() / 2, by = ex.button_y() / 2;
at(cA, "pointerdown", bx, by, { button: 0, pointerId: 2 });
at(cA, "pointerup", bx, by, { button: 0 });
frame();
expect("button clicked through the canvas", ex.clicks() === 1);
expect("focus moved to A", document.activeElement === cA && ex.win_focused(0) === 1 && ex.win_focused(1) === 0 && ex.kind_count(11) === 1);
at(cA, "pointerdown", ex.input_x() / 2, ex.input_y() / 2, { button: 0, pointerId: 3 });
at(cA, "pointerup", ex.input_x() / 2, ex.input_y() / 2, { button: 0 });
frame();
for (const ch of ["h", "é", "!"]) key(cA, { key: ch, code: ch === "h" ? "KeyH" : "" });
frame();
str("hé!");
expect("typed into the text field", ex.input_is() === 1, `len ${ex.input_len()}`);
// clipboard: the program copies
expect("clipboard set -> navigator.clipboard", ex.clip_set() === 1 && window.clipboardWrites.at(-1) === "copied ✓");
str("copied ✓");
expect("clipboard get after set", ex.clip_is() === 1);
// paste: Ctrl+V waits for the paste event, which brings the text
const kv = new Event("keydown", { key: "v", code: "KeyV", ctrlKey: true });
cA.dispatchEvent(kv);
expect("Ctrl+V keeps its default (the paste event)", !kv.defaultPrevented);
const paste = new Event("paste", { clipboardData: { getData: (t) => (t === "text/plain" ? " pasted ü" : "") } });
document.dispatchEvent(paste);
cA.dispatchEvent(new Event("keyup", { key: "v", code: "KeyV", ctrlKey: true }));
frame();
str("hé! pasted ü");
expect("Ctrl+V pastes the paste event's text", ex.input_is() === 1, `len ${ex.input_len()}`);
// a held Ctrl+V with no paste event still arrives (100 ms)
cA.dispatchEvent(new Event("keydown", { key: "v", code: "KeyV", ctrlKey: true }));
await new Promise((r) => setTimeout(r, 150));
frame();
str("hé! pasted ü pasted ü");
expect("Ctrl+V without a paste event: last known text", ex.input_is() === 1, `len ${ex.input_len()}`);
// copy event: the program's copy reaches the system clipboard through it
cA.dispatchEvent(new Event("keydown", { key: "c", code: "KeyC", ctrlKey: true }));
// (the text field copies nothing without a selection)
let set = null;
const copy = new Event("copy", { clipboardData: { setData: (t, v) => { set = v; } } });
document.dispatchEvent(copy);
expect("copy event untouched when the program copied nothing", set === null && !copy.defaultPrevented);
cA.dispatchEvent(new Event("keydown", { key: "a", code: "KeyA", ctrlKey: true }));
cA.dispatchEvent(new Event("keydown", { key: "c", code: "KeyC", ctrlKey: true }));
document.dispatchEvent(copy);
frame();
expect("Ctrl+A, Ctrl+C: the copy event carries the selection", set === "hé! pasted ü pasted ü" && copy.defaultPrevented, String(set));

// sizes, scale
ex.set_size(0, 220, 160);
frame();
expect("set_size: CSS + backing store, buffer resized", cA.style.width === "220px" && cA.width === 440 && ex.win_w(0) === 440 && ex.win_h(0) === 320);
fitCanvas.cssBox = [320, 240];
window.resizeObservers.forEach((o) => o.fire());
frame();
expect("fit canvas follows the page (ResizeObserver)", ex.win_w(2) === 320 && ex.win_h(2) === 240 && cC.width === 320);
globals.devicePixelRatio = 1;   // (globals is a copy of the window's properties)
window.fireMediaChange();
frame();
expect("devicePixelRatio change: SCALE", ex.win_scale1000(0) === 1000 && ex.win_w(0) === 220 && cA.width === 220 && ex.kind_count(15) === 1 && ex.win_scale1000(2) === 1000);
expect("re-presented at the new scale", JSON.stringify(cB.readPixel(2, 2)) === "[255,0,0,255]" && cB.width === 120);
ex.set_cursor(0, 2);
expect("cursor: CSS cursor", cA.style.cursor === "pointer");
ex.set_title(1);
expect("set_title", cB.getAttribute("aria-label") === "Renamed ✓");
// timers and the clock
expect("set_timeout fired from the frame loop", ex.timer_fired() === 1);
// closing, quitting
ex.close_b();
frame();
expect("request_close closes the window", ex.window_count() === 2 && app.gui.canvas(ex.win_id(1)) === null);
ex.quit();
frame();
expect("quit(): the frame loop stops", !window.hasFrame() && !app.gui.running);

console.log(failures === 0 ? "wasm gui: all passed" : `wasm gui: ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
