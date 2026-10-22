// Drives the std/wasm DOM test module (build/wasm_dom_tests, built by
// run.sh) against fake_dom.mjs through the real std/wasm/safec_dom.js glue.
//   usage: node run.mjs <module.wasm> <path/to/safec_dom.js>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { createFakeWindow, Event } from "./fake_dom.mjs";

const [wasmPath, gluePath] = process.argv.slice(2);
const { loadSafeC } = await import(pathToFileURL(gluePath).href);
const window = createFakeWindow();
const { document } = window;
const logs = [];
const fakeConsole = {
    debug: (m) => logs.push(["debug", m]), info: (m) => logs.push(["info", m]),
    log: (m) => logs.push(["log", m]), warn: (m) => logs.push(["warn", m]),
    error: (m) => { logs.push(["error", m]); console.error(m); },
};
window.routes.set("/api/data", (init) => ({
    status: 201, headers: { "content-type": "text/plain" },
    text: `echo:${init.body}|${init.headers["Content-Type"]}|${init.headers["X-A"]}`,
}));

// Like a browser, the window's functions only accept the window itself
// (or undefined) as their receiver; the glue must not call them through
// a copy of the globals object.
const globals = { ...window, window, console: fakeConsole };
for (const name of ["fetch", "setTimeout", "clearTimeout", "setInterval",
                    "clearInterval", "requestAnimationFrame"]) {
    const fn = globals[name];
    if (typeof fn !== "function") continue;
    globals[name] = function (...args) {
        if (this !== undefined && this !== globals && this !== globalThis)
            throw new TypeError(`Illegal invocation: ${name}`);
        return fn.apply(window, args);
    };
}
const app = await loadSafeC(readFileSync(wasmPath), { globals });
const ex = app.exports;
let failures = 0;
const step = (name, fn) => {
    const n = fn();
    console.log(`  ${n === 0 ? "PASS" : "FAIL"}  ${name}${n ? ` (${n} failed checks)` : ""}`);
    if (n !== 0) failures++;
};
const expect = (name, cond) => {
    console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
    if (!cond) failures++;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

step("elements, content, attributes, classes, properties", () => ex.t_elements());
const list = document.getElementById("list");
expect("tree built in the document", list && list.children.length === 3 && list.textContent === "first01");
expect("style + visibility reach the element", document.querySelector("div").style.getPropertyValue("background-color") === "red"
    && document.querySelector("div").style.display === "none");
expect("focus() called", document.activeElement === document.getElementById("name"));
expect("console level routed", logs.some(([l, m]) => l === "info" && m === "elements done"));
expect("document title set", document.title === "SafeC");

step("event listeners registered", () => ex.t_events_setup());
const btn = document.getElementById("btn");
const c1 = new Event("click", { clientX: 7, altKey: true });
btn.dispatchEvent(c1);
expect("prevent_default reaches the event", c1.defaultPrevented);
btn.dispatchEvent(new Event("click", { clientX: 7, altKey: true }));
window.dispatchEvent(new Event("keydown", { key: "A", code: "KeyA", shiftKey: true }));
const name = document.getElementById("name");
name.value = "typed text";
name.dispatchEvent(new Event("input"));
step("event payloads (target, coords, modifiers, key, value)", () => ex.t_events_check());
btn.dispatchEvent(new Event("click"));
expect("dom_off removes the listener", ex.clicks() === 2);

step("storage, history, location (sync part)", () => ex.t_async_setup());
expect("navigate assigns location", window.location.assigned === "/elsewhere");
// Wait for every scheduled callback (bounded), not a fixed time: a loaded
// CI runner can take longer than any fixed sleep. A short extra wait then
// gives the cancelled timer/frame time to (wrongly) fire.
for (let waited = 0; ex.t_async_pending() && waited < 2000; waited += 10) await sleep(10);
await sleep(20);
step("timers, animation frames, fetch, popstate", () => ex.t_async_check());

step("signals bound to the DOM", () => ex.t_hydrate());
const field = document.getElementById("field");
field.value = "Grace";
field.dispatchEvent(new Event("input"));
const box = document.getElementById("box");
box.checked = false;
box.dispatchEvent(new Event("change"));
expect("two-way: class follows the checkbox", !document.querySelector("div").classList.contains("on"));
step("two-way bindings and String signals", () => ex.t_hydrate_check());
expect("text binding updated from SafeC", document.querySelector("h1").textContent === "Bye");

console.log(`--- wasm dom: ${failures === 0 ? "all passed" : failures + " failed"} ---`);
process.exit(failures === 0 ? 0 : 1);
