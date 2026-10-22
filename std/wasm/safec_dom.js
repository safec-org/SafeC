// SafeC Standard Library — JavaScript glue for std/wasm (dom.h).
//
// Implements every env.js_* import std/wasm/dom.h declares, plus the
// handful of libc-shaped symbols a freestanding module may still import.
// An ES module with no dependencies; works in browsers and in Node (pass
// your own `document`/`window`, e.g. a fake DOM, through `options.globals`).
//
//   import { loadSafeC } from "./safec_dom.js";
//   const app = await loadSafeC(fetch("/app.wasm"));
//   app.exports.start();                     // your export_name functions
//
//   // Lower level: build the import object yourself.
//   const glue = createSafeCGlue({ globals: { document, window } });
//   const { instance } = await WebAssembly.instantiate(bytes, glue.imports);
//   glue.attach(instance);
//
// loadSafeC(source, options):
//   source   a Response, a Promise of one, an ArrayBuffer/typed array, or a
//            WebAssembly.Module
//   options  { globals, imports, onLog, prefix }
//     globals  where to find document, window, localStorage,
//              sessionStorage, history, location, fetch, setTimeout,
//              clearTimeout, setInterval, clearInterval,
//              requestAnimationFrame, cancelAnimationFrame, performance,
//              console (each defaults to globalThis's)
//     imports  extra import entries, merged into env (yours win)
//     prefix   text prepended to console output (default "")
// Returns { instance, exports, glue }.
//
// Handles: integers into an element table (-1 = none, -2 = document,
// -3 = window). glue.handle(el) / glue.element(h) convert by hand.

export function createSafeCGlue(options = {}) {
    const g = Object.assign({}, globalThis, options.globals || {});
    const prefix = options.prefix || "";
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    let instance = null;

    // ── memory helpers ────────────────────────────────────────────────────
    const mem = () => new Uint8Array(instance.exports.memory.buffer);
    const str = (ptr, len) => {
        len = Number(len);
        if (len <= 0 || ptr === 0 || ptr === 0n) return "";
        return decoder.decode(mem().subarray(Number(ptr), Number(ptr) + len));
    };
    const optStr = (ptr, len) => (Number(len) < 0 ? null : str(ptr, len));
    // Copies up to cap bytes of `s`'s UTF-8 into ptr; returns the full byte
    // length, or -1 when s is null/undefined.
    const put = (s, ptr, cap) => {
        if (s === null || s === undefined) return -1;
        const bytes = encoder.encode(String(s));
        const n = Math.min(bytes.length, Number(cap));
        if (n > 0) mem().set(bytes.subarray(0, n), Number(ptr));
        return bytes.length;
    };
    const i32 = () => new Int32Array(instance.exports.memory.buffer);

    // ── handle table ──────────────────────────────────────────────────────
    const table = [];
    const free = [];
    const ids = new Map();
    const handle = (el) => {
        if (el === null || el === undefined) return -1;
        if (el === g.document) return -2;
        if (el === g.window) return -3;
        if (ids.has(el)) return ids.get(el);
        const h = free.length ? free.pop() : table.length;
        table[h] = el;
        ids.set(el, h);
        return h;
    };
    const element = (h) => {
        if (h === -2) return g.document;
        if (h === -3) return g.window;
        return h >= 0 ? table[h] : undefined;
    };
    const release = (h) => {
        if (h < 0 || table[h] === undefined) return;
        ids.delete(table[h]);
        table[h] = undefined;
        free.push(h);
    };
    const putHandles = (list, out, cap) => {
        const n = list.length;
        const view = i32();
        const base = Number(out) >> 2;
        for (let i = 0; i < Math.min(n, cap); i++) view[base + i] = handle(list[i]);
        return n;
    };

    // ── callbacks back into the module ────────────────────────────────────
    let currentEvent = null;
    const dispatch = (id, ev) => {
        const prev = currentEvent;
        currentEvent = ev;
        try { instance.exports.std_dom_dispatch_event(id); }
        finally { currentEvent = prev; }
    };
    const listeners = new Map();
    let nextListener = 1;
    const timers = new Map();
    let nextTimer = 1;
    const frames = new Map();
    let nextFrame = 1;
    const fetches = new Map();
    let nextFetch = 1;

    const storage = (kind) => (kind === 1 ? g.sessionStorage : g.localStorage);
    const consoleFn = (level) => {
        const c = g.console || console;
        return [c.debug, c.info, c.log, c.warn, c.error][level] || c.log;
    };
    const parseHeaders = (text) => {
        const h = {};
        for (const line of text.split(/\r?\n/)) {
            const i = line.indexOf(":");
            if (i > 0) h[line.slice(0, i).trim()] = line.slice(i + 1).trim();
        }
        return h;
    };

    const env = {
        // libc-shaped symbols a freestanding module may import (std's
        // checked allocation paths and Vec::sort reference them).
        qsort: () => {},
        abort: () => { throw new Error("wasm module called abort()"); },
        fprintf: () => 0,
        printf: () => 0,
        __stderrp: new WebAssembly.Global({ value: "i32", mutable: false }, 0),
        // Fallbacks for libc string functions std code may call when the
        // module's own runtime doesn't define them (a definition inside the
        // module always wins: then these are simply not imported).
        // compiler-rt's 128-bit multiply (overflow-checked size math), as
        // wasm32 lowers it: result pointer, then each operand as two i64s.
        __multi3: (out, alo, ahi, blo, bhi) => {
            const M = (1n << 64n) - 1n;
            const a = (BigInt.asUintN(64, ahi) << 64n) | BigInt.asUintN(64, alo);
            const b = (BigInt.asUintN(64, bhi) << 64n) | BigInt.asUintN(64, blo);
            const r = BigInt.asUintN(128, a * b);
            const v = new DataView(instance.exports.memory.buffer);
            v.setBigUint64(Number(out), r & M, true);
            v.setBigUint64(Number(out) + 8, r >> 64n, true);
        },
        strlen: (p) => { const m = mem(); let i = Number(p); const s = i; while (m[i] !== 0) i++; return i - s; },

        // ── original surface ──
        js_get_element: (p, n) => handle(g.document.querySelector(str(p, n))),
        js_set_text: (h, p, n) => { element(h).textContent = str(p, n); },
        js_set_attr: (h, np, nn, vp, vn) => { element(h).setAttribute(str(np, nn), str(vp, vn)); },
        js_add_click_listener: (h, id) => {
            element(h).addEventListener("click", (ev) => dispatch(id, ev));
            return 0;
        },
        js_console_log: (p, n) => { consoleFn(2).call(g.console || console, prefix + str(p, n)); },

        // ── elements ──
        js_query: (parent, p, n) => {
            const root = parent === -1 ? g.document : element(parent);
            return root ? handle(root.querySelector(str(p, n))) : -1;
        },
        js_query_all: (parent, p, n, out, cap) => {
            const root = parent === -1 ? g.document : element(parent);
            return root ? putHandles(Array.from(root.querySelectorAll(str(p, n))), out, cap) : 0;
        },
        js_create_element: (p, n) => handle(g.document.createElement(str(p, n))),
        js_create_text: (p, n) => handle(g.document.createTextNode(str(p, n))),
        js_append_child: (parent, child) => { element(parent).appendChild(element(child)); },
        js_insert_before: (parent, child, ref) => {
            element(parent).insertBefore(element(child), ref < 0 ? null : element(ref));
        },
        js_remove: (h) => { const el = element(h); if (el && el.remove) el.remove(); },
        js_release: (h) => release(h),
        js_relative: (h, which) => {
            const el = element(h);
            if (!el) return -1;
            switch (which) {
                case 0: return handle(el.parentElement || el.parentNode || null);
                case 1: return handle(el.firstElementChild || null);
                case 2: return handle(el.nextElementSibling || null);
                case 3: return handle(el.previousElementSibling || null);
                case 4: return handle(el.lastElementChild || null);
                case 10: return handle(g.document.body || null);
                case 11: return handle(g.document.head || null);
            }
            return -1;
        },
        js_children: (h, out, cap) => {
            const el = element(h);
            return el ? putHandles(Array.from(el.children || []), out, cap) : 0;
        },

        // ── content ──
        js_get_text: (h, buf, cap) => put(element(h).textContent, buf, cap),
        js_set_html: (h, p, n) => { element(h).innerHTML = str(p, n); },
        js_get_html: (h, buf, cap) => put(element(h).innerHTML, buf, cap),
        js_get_attr: (h, np, nn, buf, cap) => put(element(h).getAttribute(str(np, nn)), buf, cap),
        js_remove_attr: (h, np, nn) => { element(h).removeAttribute(str(np, nn)); },
        js_class_op: (h, op, p, n) => {
            const cl = element(h).classList;
            const c = str(p, n);
            if (op === 0) { cl.add(c); return 1; }
            if (op === 1) { cl.remove(c); return 0; }
            if (op === 2) return cl.toggle(c) ? 1 : 0;
            return cl.contains(c) ? 1 : 0;
        },
        js_set_style: (h, pp, pn, vp, vn) => {
            const v = str(vp, vn);
            const st = element(h).style;
            if (v === "") st.removeProperty(str(pp, pn)); else st.setProperty(str(pp, pn), v);
        },
        js_get_prop: (h, np, nn, buf, cap) => {
            const v = element(h)[str(np, nn)];
            return put(v === undefined || v === null ? null : v, buf, cap);
        },
        js_set_prop: (h, np, nn, vp, vn) => { element(h)[str(np, nn)] = str(vp, vn); },
        js_get_prop_num: (h, np, nn) => Number(element(h)[str(np, nn)]) || 0,
        js_set_prop_num: (h, np, nn, v) => {
            const el = element(h);
            const name = str(np, nn);
            el[name] = typeof el[name] === "boolean" ? v !== 0 : v;
        },
        js_call: (h, mp, mn) => {
            const el = element(h);
            const m = str(mp, mn);
            if (el && typeof el[m] === "function") el[m]();
        },

        // ── events ──
        js_add_listener: (h, tp, tn, id, flags) => {
            const target = element(h);
            if (!target || !target.addEventListener) return -1;
            const type = str(tp, tn);
            const opts = { capture: !!(flags & 1), once: !!(flags & 2), passive: !!(flags & 4) };
            const fn = (ev) => dispatch(id, ev);
            target.addEventListener(type, fn, opts);
            const lid = nextListener++;
            listeners.set(lid, { target, type, fn, opts });
            return lid;
        },
        js_remove_listener: (lid) => {
            const l = listeners.get(lid);
            if (!l) return;
            l.target.removeEventListener(l.type, l.fn, l.opts);
            listeners.delete(lid);
        },
        js_event_num: (field) => {
            const e = currentEvent;
            if (!e) return 0;
            switch (field) {
                case 0: return handle(e.target || null);
                case 1: return e.clientX || 0;
                case 2: return e.clientY || 0;
                case 3: return e.button || 0;
                case 4: return e.buttons || 0;
                case 5: return (e.altKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.shiftKey ? 4 : 0) | (e.metaKey ? 8 : 0);
                case 6: return e.deltaX || 0;
                case 7: return e.deltaY || 0;
                case 8: return e.repeat ? 1 : 0;
                case 9: return e.timeStamp || 0;
                case 10: return e.offsetX || 0;
                case 11: return e.offsetY || 0;
                case 12: return e.target && e.target.checked ? 1 : 0;
                case 13: return handle(e.currentTarget || null);
            }
            return 0;
        },
        js_event_str: (field, buf, cap) => {
            const e = currentEvent;
            if (!e) return -1;
            switch (field) {
                case 0: return put(e.type, buf, cap);
                case 1: return put(e.key ?? "", buf, cap);
                case 2: return put(e.code ?? "", buf, cap);
                case 3: return put(e.target && e.target.value !== undefined ? e.target.value : "", buf, cap);
                case 4: return put(e.data === undefined || e.data === null ? "" :
                                   (typeof e.data === "string" ? e.data : JSON.stringify(e.data)), buf, cap);
                case 5: return put(JSON.stringify(e.state === undefined ? null : e.state), buf, cap);
            }
            return -1;
        },
        js_event_control: (op) => {
            if (!currentEvent) return;
            if (op === 0 && currentEvent.preventDefault) currentEvent.preventDefault();
            if (op === 1 && currentEvent.stopPropagation) currentEvent.stopPropagation();
        },

        // ── timers ──
        js_set_timer: (id, ms, repeat) => {
            const tid = nextTimer++;
            const fire = () => {
                if (!repeat) timers.delete(tid);
                instance.exports.std_dom_dispatch_event(id);
            };
            timers.set(tid, { repeat, h: repeat ? g.setInterval(fire, ms) : g.setTimeout(fire, ms) });
            return tid;
        },
        js_clear_timer: (tid) => {
            const t = timers.get(tid);
            if (!t) return;
            if (t.repeat) g.clearInterval(t.h); else g.clearTimeout(t.h);
            timers.delete(tid);
        },
        js_request_frame: (id) => {
            const fid = nextFrame++;
            const raf = g.requestAnimationFrame || ((cb) => g.setTimeout(() => cb(g.performance ? g.performance.now() : Date.now()), 16));
            frames.set(fid, raf((ts) => {
                frames.delete(fid);
                instance.exports.std_dom_dispatch_frame(id, ts);
            }));
            return fid;
        },
        js_cancel_frame: (fid) => {
            if (!frames.has(fid)) return;
            const caf = g.cancelAnimationFrame || g.clearTimeout;
            caf(frames.get(fid));
            frames.delete(fid);
        },
        js_now: () => (g.performance ? g.performance.now() : Date.now()),

        // ── fetch ──
        js_fetch: (id, mp, mn, up, un, hp, hn, bp, bn) => {
            const rid = nextFetch++;
            const init = { method: str(mp, mn) };
            const headers = optStr(hp, hn);
            if (headers !== null) init.headers = parseHeaders(headers);
            const body = optStr(bp, bn);
            if (body !== null) init.body = body;
            const url = str(up, un);
            const record = { status: 0, statusText: "", url, text: "", headers: null };
            fetches.set(rid, record);
            Promise.resolve()
                .then(() => g.fetch(url, init))
                .then(async (r) => {
                    record.status = r.status;
                    record.statusText = r.statusText || "";
                    record.url = r.url || url;
                    record.headers = r.headers;
                    record.text = await r.text();
                }, (err) => { record.statusText = String(err && err.message || err); })
                .then(() => instance.exports.std_dom_dispatch_fetch(id, rid, record.status));
            return rid;
        },
        js_fetch_str: (rid, field, buf, cap) => {
            const r = fetches.get(rid);
            if (!r) return -1;
            return put(field === 0 ? r.text : field === 1 ? r.statusText : r.url, buf, cap);
        },
        js_fetch_header: (rid, np, nn, buf, cap) => {
            const r = fetches.get(rid);
            if (!r || !r.headers) return -1;
            return put(r.headers.get(str(np, nn)), buf, cap);
        },
        js_fetch_done: (rid) => { fetches.delete(rid); },

        // ── storage ──
        js_storage_get: (kind, kp, kn, buf, cap) => {
            const s = storage(kind);
            return s ? put(s.getItem(str(kp, kn)), buf, cap) : -1;
        },
        js_storage_set: (kind, kp, kn, vp, vn) => {
            try { storage(kind).setItem(str(kp, kn), str(vp, vn)); return 0; }
            catch (e) { return -1; }
        },
        js_storage_remove: (kind, kp, kn) => { const s = storage(kind); if (s) s.removeItem(str(kp, kn)); },
        js_storage_clear: (kind) => { const s = storage(kind); if (s) s.clear(); },
        js_storage_length: (kind) => { const s = storage(kind); return s ? s.length : 0; },
        js_storage_key: (kind, i, buf, cap) => { const s = storage(kind); return s ? put(s.key(i), buf, cap) : -1; },

        // ── location / history / console ──
        js_location: (field, buf, cap) => {
            const l = g.location;
            if (!l) return -1;
            return put([l.href, l.pathname, l.search, l.hash, l.host, l.origin][field], buf, cap);
        },
        js_history: (op, up, un) => {
            const url = str(up, un);
            if (op === 0) g.history.pushState(null, "", url);
            else if (op === 1) g.history.replaceState(null, "", url);
            else if (op === 2) g.history.back();
            else if (op === 3) g.history.forward();
            else if (op === 4) g.location.assign(url);
        },
        js_console: (level, p, n) => { consoleFn(level).call(g.console || console, prefix + str(p, n)); },
    };
    Object.assign(env, options.imports || {});

    return {
        imports: { env },
        attach(inst) { instance = inst; },
        handle,
        element,
        release,
        get instance() { return instance; },
    };
}

export async function loadSafeC(source, options = {}) {
    const glue = createSafeCGlue(options);
    let src = await source;
    let result;
    if (src instanceof WebAssembly.Module) {
        result = { instance: await WebAssembly.instantiate(src, glue.imports) };
    } else {
        if (src && typeof src.arrayBuffer === "function") src = await src.arrayBuffer();
        result = await WebAssembly.instantiate(src, glue.imports);
    }
    glue.attach(result.instance);
    return { instance: result.instance, exports: result.instance.exports, glue };
}
