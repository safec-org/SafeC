// fullstack_demo — browser-side glue for counter_client.wasm.
//
// Provides the env.js_* imports src/counter_client.sc's std/wasm/dom.h
// declares as 'extern' (an undefined extern in a wasm32 module linked
// with --allow-undefined becomes an import resolved here, from the
// default "env" module — see std/wasm/wasm_rt.h/dom.h's header comments
// for the full mechanism). Kept as a plain, separate static file rather
// than inlined into pages.scx's <script> tag: scx's '{expr}'/'{!expr}'
// interpolation syntax (see std/scx/scx.h) scans literal markup text for
// '{'/'}' with no raw/CDATA escape hatch, and this file is full of both
// (object literals, function bodies) — inlining it would make the scx
// transpiler misparse ordinary JS braces as template interpolations.
let wasmInstance = null;
const elements = [];      // handle -> Element
const handleOf = new Map(); // Element -> handle

function elementHandle(el) {
    if (handleOf.has(el)) return handleOf.get(el);
    const h = elements.length;
    elements.push(el);
    handleOf.set(el, h);
    return h;
}

function readString(ptr, len) {
    const bytes = new Uint8Array(wasmInstance.exports.memory.buffer, Number(ptr), Number(len));
    return new TextDecoder().decode(bytes);
}

const importObject = {
    env: {
        // Vec's checked alloc()/dealloc() path and Vec::sort() reference
        // these even though this demo never triggers them — see
        // std/wasm/wasm_rt.h's header comment on why free() etc alone
        // aren't enough to avoid every libc-shaped extern.
        qsort: () => {},
        abort: () => { throw new Error("counter_client.wasm: abort() called"); },
        fprintf: () => 0,
        __stderrp: new WebAssembly.Global({ value: "i32", mutable: false }, 0),

        js_get_element: (selPtr, selLen) => {
            const sel = readString(selPtr, selLen);
            const el = document.querySelector(sel);
            return el ? elementHandle(el) : -1;
        },
        js_set_text: (handle, textPtr, textLen) => {
            elements[handle].textContent = readString(textPtr, textLen);
        },
        js_set_attr: (handle, namePtr, nameLen, valPtr, valLen) => {
            elements[handle].setAttribute(readString(namePtr, nameLen), readString(valPtr, valLen));
        },
        js_add_click_listener: (handle, callbackId) => {
            elements[handle].addEventListener("click", () => {
                wasmInstance.exports.std_dom_dispatch_event(callbackId);
            });
            return 0;
        },
        js_console_log: (ptr, len) => {
            console.log("[counter_client.wasm]", readString(ptr, len));
        },
    },
};

const { instance } = await WebAssembly.instantiate(
    await (await fetch("/counter_client.wasm")).arrayBuffer(),
    importObject,
);
wasmInstance = instance;
instance.exports.hydrate_init();
