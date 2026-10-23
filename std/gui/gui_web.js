// SafeC Standard Library — JavaScript glue for std/gui on WebAssembly
// (gui_web.h, gui_web_wasm.sc).
//
// Implements the env.std_gui_web_* imports a wasm32 module built with
// std/gui needs — each GuiWindow a <canvas>, input events forwarded to the
// module, putImageData presentation — and drives the module's GuiApp from
// requestAnimationFrame. An ES module with no dependencies; works in
// browsers and, with stand-ins passed through `options.globals`, in Node.
//
//   import { loadSafeCGui } from "./gui_web.js";
//   const app = await loadSafeCGui(fetch("app.wasm"), { entry: "start" });
//
//   // With std/wasm's DOM glue in the same module (dom.h + gui.h):
//   import { loadSafeC } from "./safec_dom.js";
//   const app = await loadSafeCGui(fetch("app.wasm"), { loadSafeC, entry: "start" });
//
//   // Lower level: your own instantiation.
//   const gui = createSafeCGui(options);
//   const { instance } = await WebAssembly.instantiate(bytes, { env: { ...gui.imports.env } });
//   gui.attach(instance);
//   gui.start("start");            // calls the export, then runs the frame loop
//
// options:
//   canvas   where windows go: a canvas element, a CSS selector, or a
//            function (index, title) => canvas (index: 0 for the first
//            window, ...). Default: the next unused canvas[data-safec-gui]
//            of the page, else a new <canvas> appended to <body>.
//   fit      true: a canvas keeps the size the page's CSS gives it (and
//            follows it) instead of taking the window's size; per canvas
//            also with a data-safec-fit attribute.
//   title    true: the first window's title becomes document.title.
//   entry    loadSafeCGui: the export to call once instantiated (default
//            "start"; false: none — call gui.start() yourself).
//   loadSafeC  loadSafeCGui: std/wasm/safec_dom.js's loadSafeC, to
//            instantiate through it (DOM imports + these).
//   imports  extra env imports (yours win).
//   globals  where to find document, window, navigator, performance,
//            requestAnimationFrame, ImageData, ResizeObserver, matchMedia,
//            OffscreenCanvas, setTimeout (default: globalThis).
// gui.start(entry) returns the entry's result; gui.stop() ends the frame
// loop; gui.canvas(id) is a window's canvas (id = GuiWindow.id).

const EV = { MOUSE_DOWN: 3, MOUSE_UP: 4, MOUSE_MOVE: 5, KEY_DOWN: 6, KEY_UP: 7, CHAR: 8, SCROLL: 9,
             FOCUS_IN: 10, FOCUS_OUT: 11, MOUSE_ENTER: 13, MOUSE_LEAVE: 14, SCALE: 15, RESIZE: 2 };
const MOD = { SHIFT: 1, CTRL: 2, ALT: 4, META: 8 };
const CURSORS = ["default", "text", "pointer", "crosshair", "ew-resize", "ns-resize", "nwse-resize",
                 "nesw-resize", "move", "wait", "not-allowed", "none"];
const NAMED_KEYS = {
    ArrowLeft: 1, ArrowRight: 2, ArrowUp: 3, ArrowDown: 4, Enter: 5, NumpadEnter: 5, Escape: 6, Tab: 7,
    Backspace: 8, Space: 9, Delete: 10, Home: 11, End: 12, PageUp: 13, PageDown: 14, Insert: 15,
    ShiftLeft: 28, ShiftRight: 28, ControlLeft: 29, ControlRight: 29, AltLeft: 30, AltRight: 30,
    MetaLeft: 31, MetaRight: 31, OSLeft: 31, OSRight: 31, ContextMenu: 32,
};

// GUI_KEY_* for a KeyboardEvent: by physical key (code), like the native
// backends' unshifted keysyms; letters are 'A'..'Z', digits '0'..'9'.
function keyCodeOf(e) {
    const c = e.code || "";
    if (c in NAMED_KEYS) return NAMED_KEYS[c];
    let m = /^Key([A-Z])$/.exec(c);
    if (m) return m[1].charCodeAt(0);
    m = /^(?:Digit|Numpad)([0-9])$/.exec(c);
    if (m) return 48 + Number(m[1]);
    m = /^F([0-9]{1,2})$/.exec(c);
    if (m && Number(m[1]) >= 1 && Number(m[1]) <= 12) return 15 + Number(m[1]);
    // no code (synthetic events): fall back to the key
    const k = e.key || "";
    if (k in NAMED_KEYS) return NAMED_KEYS[k];
    if (k === " ") return 9;
    if (/^[a-zA-Z]$/.test(k)) return k.toUpperCase().charCodeAt(0);
    if (/^[0-9]$/.test(k)) return k.charCodeAt(0);
    return 0;
}

// printf-style formatting of a wasm32 varargs call: 'va' points at the
// arguments, each int/pointer 4 bytes, long long/double 8 (8-aligned).
function formatC(view, fmt, va) {
    let out = "", i = 0, p = va;
    const take4 = () => { p = (p + 3) & ~3; const v = view.getInt32(p, true); p += 4; return v; };
    const take8i = () => { p = (p + 7) & ~7; const v = view.getBigInt64(p, true); p += 8; return v; };
    const take8f = () => { p = (p + 7) & ~7; const v = view.getFloat64(p, true); p += 8; return v; };
    while (i < fmt.length) {
        const ch = fmt[i++];
        if (ch !== "%") { out += ch; continue; }
        const m = /^([-+ 0#]*)(\*|\d+)?(?:\.(\*|\d+))?(hh|h|ll|l|z|j|t|L)?([diuxXofFeEgGcsp%])/.exec(fmt.slice(i));
        if (!m) { out += "%"; continue; }
        i += m[0].length;
        const flags = m[1];
        let width = m[2] === "*" ? take4() : m[2] ? Number(m[2]) : 0;
        let prec = m[3] === "*" ? take4() : m[3] !== undefined ? Number(m[3]) : -1;
        const len = m[4] || "", conv = m[5];
        let s;
        if (conv === "%") { out += "%"; continue; }
        if ("diuxXo".includes(conv)) {
            let v = len === "ll" || len === "j" ? take8i() : BigInt(take4());
            const signed = conv === "d" || conv === "i";
            if (!signed) v = BigInt.asUintN(len === "ll" || len === "j" ? 64 : 32, v);
            const neg = v < 0n;
            if (neg) v = -v;
            s = v.toString(conv === "o" ? 8 : conv === "x" || conv === "X" ? 16 : 10);
            if (conv === "X") s = s.toUpperCase();
            if (prec >= 0) s = s.padStart(prec, "0");
            if (neg) s = "-" + s; else if (signed && flags.includes("+")) s = "+" + s; else if (signed && flags.includes(" ")) s = " " + s;
        } else if ("fFeEgG".includes(conv)) {
            const v = take8f();
            const pr = prec < 0 ? 6 : prec;
            s = conv === "f" || conv === "F" ? v.toFixed(pr) : conv === "e" || conv === "E" ? v.toExponential(pr) : String(Number(v.toPrecision(pr || 1)));
            if (conv === "E" || conv === "G") s = s.toUpperCase();
            if (v >= 0 && flags.includes("+")) s = "+" + s;
        } else if (conv === "c") {
            s = String.fromCharCode(take4() & 0xFF);
        } else if (conv === "s") {
            const ptr = take4() >>> 0;
            let e = ptr;
            const u8 = new Uint8Array(view.buffer);
            while (u8[e] !== 0 && (prec < 0 || e - ptr < prec)) e++;
            s = ptr ? new TextDecoder().decode(u8.subarray(ptr, e)) : "(null)";
        } else {
            s = "0x" + (take4() >>> 0).toString(16);
        }
        if (s.length < width) {
            const zero = flags.includes("0") && !flags.includes("-") && conv !== "s" && conv !== "c";
            if (flags.includes("-")) s = s.padEnd(width);
            else if (zero) { const sign = /^[-+ ]/.test(s) ? s[0] : ""; s = sign + s.slice(sign.length).padStart(width - sign.length, "0"); }
            else s = s.padStart(width);
        }
        out += s;
    }
    return out;
}

// Libc/libm-shaped imports a freestanding module may still have (std's
// file and environment calls, which find nothing in a browser; the printf
// family; libm): a definition inside the module always wins — then these
// simply aren't imported.
function libcFallbacks(getInstance, log) {
    const view = () => new DataView(getInstance().exports.memory.buffer);
    const cstr = (p) => { const u8 = new Uint8Array(getInstance().exports.memory.buffer); let e = p; while (u8[e] !== 0) e++; return new TextDecoder().decode(u8.subarray(p, e)); };
    const encoder = new TextEncoder();
    const snprintf = (buf, n, fmt, va) => {
        const bytes = encoder.encode(formatC(view(), cstr(fmt), va));
        if (n > 0) {
            const k = Math.min(bytes.length, n - 1);
            const u8 = new Uint8Array(getInstance().exports.memory.buffer);
            u8.set(bytes.subarray(0, k), buf);
            u8[buf + k] = 0;
        }
        return bytes.length;
    };
    let line = "";
    const emit = (text) => {
        line += text;
        let nl;
        while ((nl = line.indexOf("\n")) >= 0) { log(line.slice(0, nl)); line = line.slice(nl + 1); }
        return encoder.encode(text).length;
    };
    const m = (f) => f;
    return {
        abort: () => { throw new Error("wasm module called abort()"); },
        snprintf, vsnprintf: snprintf,
        sprintf: (buf, fmt, va) => snprintf(buf, 0x7FFFFFFF, fmt, va),
        printf: (fmt, va) => emit(formatC(view(), cstr(fmt), va)),
        fprintf: (f, fmt, va) => emit(formatC(view(), cstr(fmt), va)),
        puts: (s) => emit(cstr(s) + "\n"),
        qsort: () => {},
        __stderrp: new WebAssembly.Global({ value: "i32", mutable: false }, 0),
        __multi3: (out, alo, ahi, blo, bhi) => {
            const M = (1n << 64n) - 1n;
            const a = (BigInt.asUintN(64, ahi) << 64n) | BigInt.asUintN(64, alo);
            const b = (BigInt.asUintN(64, bhi) << 64n) | BigInt.asUintN(64, blo);
            const r = BigInt.asUintN(128, a * b);
            const v = view();
            v.setBigUint64(Number(out), r & M, true);
            v.setBigUint64(Number(out) + 8, r >> 64n, true);
        },
        getenv: () => 0,
        fopen: () => 0, fclose: () => 0, fread: () => 0, fwrite: () => 0, ferror: () => 1,
        fseek: () => -1, ftell: () => -1, fflush: () => 0,
        sin: m(Math.sin), cos: m(Math.cos), tan: m(Math.tan), asin: m(Math.asin), acos: m(Math.acos),
        atan: m(Math.atan), atan2: m(Math.atan2), exp: m(Math.exp), log: m(Math.log), log2: m(Math.log2),
        log10: m(Math.log10), pow: m(Math.pow), cbrt: m(Math.cbrt), hypot: m(Math.hypot),
        sqrt: m(Math.sqrt), floor: m(Math.floor), ceil: m(Math.ceil), fabs: m(Math.abs), trunc: m(Math.trunc),
        sinf: m(Math.sin), cosf: m(Math.cos), tanf: m(Math.tan), atan2f: m(Math.atan2), expf: m(Math.exp),
        logf: m(Math.log), powf: (a, b) => Math.fround(Math.pow(a, b)), acosf: m(Math.acos), asinf: m(Math.asin),
        atanf: m(Math.atan), sqrtf: (x) => Math.fround(Math.sqrt(x)), floorf: m(Math.floor), ceilf: m(Math.ceil),
        fabsf: m(Math.abs), truncf: m(Math.trunc), fmod: (a, b) => a % b, fmodf: (a, b) => Math.fround(a % b),
        round: (x) => (x < 0 ? -Math.round(-x) : Math.round(x)), roundf: (x) => Math.fround(x < 0 ? -Math.round(-x) : Math.round(x)),
    };
}

export function createSafeCGui(options = {}) {
    const own = options.globals || {};
    const g = new Proxy({}, {
        get(_, k) {
            const from = k in own ? own : globalThis;
            const v = from[k];
            return typeof v === "function" && !/^[A-Z]/.test(String(k)) ? v.bind(from) : v;
        },
        has(_, k) { return k in own || k in globalThis; },
    });
    let instance = null;
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const mem = () => instance.exports.memory.buffer;
    const str = (p, n) => (n > 0 ? decoder.decode(new Uint8Array(mem(), p, n)) : "");
    const wins = new Map();          // id -> window record
    let created = 0;
    let clipText = null;             // what gui_clipboard_get_text() sees
    let clipWritten = false;         // set by clipboard_set during an event
    let heldPaste = null;            // a Ctrl/Cmd+V keydown waiting for its paste event
    let running = false;
    let rafHandle = null;
    const apple = /Mac|iPhone|iPad|iPod/.test((g.navigator && (g.navigator.platform || g.navigator.userAgent)) || "");

    const send = (w, kind, f = {}) => {
        if (!instance || !wins.has(w.id)) return;
        clipWritten = false;
        instance.exports.std_gui_web_event(w.id, kind, f.x || 0, f.y || 0, f.button || 0, f.keycode || 0,
            f.codepoint || 0, f.mods || 0, f.dx || 0, f.dy || 0, f.width || 0, f.height || 0, f.scale || 0);
    };
    const modsOf = (e) => (e.shiftKey ? MOD.SHIFT : 0) | (e.ctrlKey ? MOD.CTRL : 0) |
                          (e.altKey ? MOD.ALT : 0) | (e.metaKey ? MOD.META : 0);
    // Event position in buffer pixels.
    const posOf = (w, e) => {
        const r = w.canvas.getBoundingClientRect ? w.canvas.getBoundingClientRect() : { left: 0, top: 0, width: w.lw, height: w.lh };
        const sx = r.width > 0 ? w.canvas.width / r.width : w.scale;
        const sy = r.height > 0 ? w.canvas.height / r.height : w.scale;
        return { x: ((e.clientX ?? 0) - r.left) * sx, y: ((e.clientY ?? 0) - r.top) * sy };
    };
    const dprOf = (w) => (w.hiDpi ? (g.devicePixelRatio || 1) : 1);

    // Resizes the backing store to logical size * scale and tells the module.
    const relayout = (w, kind) => {
        if (w.fit) {
            const cw = w.canvas.clientWidth, ch = w.canvas.clientHeight;
            if (cw > 0 && ch > 0) { w.lw = cw; w.lh = ch; }
        }
        const scale = dprOf(w);
        const bw = Math.max(1, Math.round(w.lw * scale)), bh = Math.max(1, Math.round(w.lh * scale));
        if (bw === w.canvas.width && bh === w.canvas.height && scale === w.scale) return;
        const scaleChanged = scale !== w.scale;
        w.scale = scale;
        w.canvas.width = bw;
        w.canvas.height = bh;
        send(w, scaleChanged ? EV.SCALE : kind, { width: bw, height: bh, scale });
    };

    const pickCanvas = (index, title) => {
        const want = options.canvas;
        if (typeof want === "function") return want(index, title);
        if (want && typeof want === "object" && index === 0) return want;
        if (typeof want === "string" && index === 0 && g.document) return g.document.querySelector(want);
        if (g.document) {
            const free = Array.from(g.document.querySelectorAll("canvas[data-safec-gui]") || [])
                .find((c) => !c.__safecGui);
            if (free) return free;
            const c = g.document.createElement("canvas");
            (g.document.body || g.document.documentElement).appendChild(c);
            return c;
        }
        if (g.OffscreenCanvas) return new g.OffscreenCanvas(1, 1);
        return null;
    };

    const listen = (w) => {
        const c = w.canvas;
        if (!c.addEventListener) return;
        const on = (t, f, o) => { c.addEventListener(t, f, o); w.off.push(() => c.removeEventListener(t, f, o)); };
        on("pointerdown", (e) => {
            if (c.focus) c.focus();
            if (c.setPointerCapture && e.pointerId !== undefined) { try { c.setPointerCapture(e.pointerId); } catch (_) {} }
            const p = posOf(w, e);
            send(w, EV.MOUSE_DOWN, { ...p, button: [0, 2, 1][e.button] ?? 0, mods: modsOf(e) });
            if (e.preventDefault) e.preventDefault();
        });
        on("pointerup", (e) => {
            const p = posOf(w, e);
            send(w, EV.MOUSE_UP, { ...p, button: [0, 2, 1][e.button] ?? 0, mods: modsOf(e) });
        });
        on("pointermove", (e) => send(w, EV.MOUSE_MOVE, { ...posOf(w, e), mods: modsOf(e) }));
        on("pointerenter", (e) => send(w, EV.MOUSE_ENTER, posOf(w, e)));
        on("pointerleave", (e) => send(w, EV.MOUSE_LEAVE, posOf(w, e)));
        on("wheel", (e) => {
            const unit = e.deltaMode === 1 ? 1 / 3 : e.deltaMode === 2 ? 3 : 1 / 100;   // lines per unit
            send(w, EV.SCROLL, { ...posOf(w, e), dx: (e.deltaX || 0) * unit, dy: -(e.deltaY || 0) * unit, mods: modsOf(e) });
            if (e.preventDefault) e.preventDefault();
        }, { passive: false });
        on("contextmenu", (e) => { if (e.preventDefault) e.preventDefault(); });
        on("focus", () => send(w, EV.FOCUS_IN));
        on("blur", () => send(w, EV.FOCUS_OUT));
        on("keydown", (e) => {
            const mods = modsOf(e);
            const primary = apple ? e.metaKey : e.ctrlKey;
            const key = keyCodeOf(e);
            const chars = Array.from(e.key || "");
            const typed = chars.length === 1 && !e.ctrlKey && !e.metaKey ? chars[0].codePointAt(0) : 0;
            const ev = { keycode: key, codepoint: typed >= 32 && typed !== 127 ? typed : 0, mods };
            const clip = primary && (e.code === "KeyC" || e.code === "KeyX" || e.code === "KeyV");
            if (primary && e.code === "KeyV") {
                // wait for the paste event (it carries the text), at most 100 ms
                heldPaste = { w, ev, timer: g.setTimeout(() => { const h = heldPaste; heldPaste = null; if (h) send(h.w, EV.KEY_DOWN, h.ev); }, 100) };
                return;
            }
            send(w, EV.KEY_DOWN, ev);
            if (clip) {
                // Copy/cut: let the program handle the shortcut now, so its
                // text is ready for the copy event that follows this keydown.
                if (running) gui.frame();
                w.copyPending = clipWritten;
            } else if ((key !== 0 || ev.codepoint !== 0) && e.preventDefault) {
                e.preventDefault();
            }
        });
        on("keyup", (e) => {
            send(w, EV.KEY_UP, { keycode: keyCodeOf(e), mods: modsOf(e) });
        });
    };

    const docListeners = () => {
        if (!g.document || !g.document.addEventListener || docListeners.done) return;
        docListeners.done = true;
        g.document.addEventListener("paste", (e) => {
            const t = e.clipboardData && e.clipboardData.getData ? e.clipboardData.getData("text/plain") : null;
            if (t !== null && t !== undefined) clipText = String(t);
            if (heldPaste) {
                const h = heldPaste;
                heldPaste = null;
                if (g.clearTimeout) g.clearTimeout(h.timer);
                send(h.w, EV.KEY_DOWN, h.ev);
                if (e.preventDefault) e.preventDefault();
            }
        });
        const copy = (e) => {
            // the program copied during this shortcut: put its text on the
            // system clipboard through the event (works without permissions)
            let any = false;
            for (const w of wins.values()) if (w.copyPending) { any = true; w.copyPending = false; }
            if (any && clipText !== null && e.clipboardData && e.clipboardData.setData) {
                e.clipboardData.setData("text/plain", clipText);
                if (e.preventDefault) e.preventDefault();
            }
        };
        g.document.addEventListener("copy", copy);
        g.document.addEventListener("cut", copy);
    };

    const watchScale = () => {
        if (!g.matchMedia || watchScale.mq) return;
        const dpr = g.devicePixelRatio || 1;
        const mq = g.matchMedia(`(resolution: ${dpr}dppx)`);
        if (!mq || !mq.addEventListener) return;
        watchScale.mq = mq;
        mq.addEventListener("change", () => {
            watchScale.mq = null;
            for (const w of wins.values()) relayout(w, EV.SCALE);
            watchScale();
        }, { once: true });
    };

    const env = {
        std_gui_web_now: () => (g.performance ? g.performance.now() : Date.now()),
        std_gui_web_dpr: () => g.devicePixelRatio || 1,
        std_gui_web_primary_mod: () => (apple ? MOD.META : MOD.CTRL),
        std_gui_web_create: (id, tp, tn, w, h, hiDpi, visible, out) => {
            const title = str(tp, tn);
            const canvas = pickCanvas(created, title);
            if (!canvas) return 0;
            if (created === 0 && options.title && g.document) g.document.title = title;
            created++;
            canvas.__safecGui = true;
            const rec = { id, canvas, lw: w, lh: h, hiDpi: !!hiDpi, scale: 0, off: [], ctx: null, image: null,
                          fit: !!(options.fit || (canvas.hasAttribute && canvas.hasAttribute("data-safec-fit"))) };
            rec.scale = dprOf(rec);
            if (rec.fit && canvas.clientWidth > 0 && canvas.clientHeight > 0) {
                rec.lw = canvas.clientWidth;
                rec.lh = canvas.clientHeight;
            } else if (canvas.style) {
                canvas.style.width = `${w}px`;
                canvas.style.height = `${h}px`;
            }
            canvas.width = Math.max(1, Math.round(rec.lw * rec.scale));
            canvas.height = Math.max(1, Math.round(rec.lh * rec.scale));
            rec.ctx = canvas.getContext("2d");
            if (!rec.ctx) return 0;
            if (canvas.setAttribute) {
                canvas.setAttribute("tabindex", "0");
                canvas.setAttribute("aria-label", title);
            }
            if (canvas.style) {
                canvas.style.touchAction = "none";
                if (!visible) canvas.style.display = "none";
            }
            wins.set(id, rec);
            listen(rec);
            docListeners();
            watchScale();
            if (rec.fit && g.ResizeObserver) {
                rec.ro = new g.ResizeObserver(() => relayout(rec, EV.RESIZE));
                rec.ro.observe(canvas);
            }
            const v = new Int32Array(mem(), out, 3);
            v[0] = rec.lw; v[1] = rec.lh; v[2] = Math.round(rec.scale * 1000);
            return 1;
        },
        std_gui_web_destroy: (id) => {
            const w = wins.get(id);
            if (!w) return;
            for (const f of w.off) f();
            if (w.ro) w.ro.disconnect();
            w.canvas.__safecGui = false;
            wins.delete(id);
        },
        std_gui_web_present: (id, ptr, bw, bh, x, y, rw, rh) => {
            const w = wins.get(id);
            if (!w) return;
            const buf = mem();
            if (!w.image || w.imageBuf !== buf || w.imagePtr !== ptr || w.image.width !== bw || w.image.height !== bh) {
                w.image = new g.ImageData(new Uint8ClampedArray(buf, ptr, bw * bh * 4), bw, bh);
                w.imageBuf = buf;
                w.imagePtr = ptr;
            }
            w.ctx.putImageData(w.image, 0, 0, x, y, rw, rh);
            w.presents = (w.presents || 0) + 1;
        },
        std_gui_web_set_title: (id, tp, tn) => {
            const w = wins.get(id);
            if (!w) return;
            const t = str(tp, tn);
            if (w.canvas.setAttribute) w.canvas.setAttribute("aria-label", t);
            if (options.title && g.document && wins.keys().next().value === id) g.document.title = t;
        },
        std_gui_web_set_size: (id, lw, lh) => {
            const w = wins.get(id);
            if (!w) return;
            w.lw = lw;
            w.lh = lh;
            if (w.canvas.style) { w.canvas.style.width = `${lw}px`; w.canvas.style.height = `${lh}px`; }
            w.canvas.width = Math.max(1, Math.round(lw * w.scale));
            w.canvas.height = Math.max(1, Math.round(lh * w.scale));
        },
        std_gui_web_set_visible: (id, v) => {
            const w = wins.get(id);
            if (w && w.canvas.style) w.canvas.style.display = v ? "" : "none";
        },
        std_gui_web_focus: (id) => { const w = wins.get(id); if (w && w.canvas.focus) w.canvas.focus(); },
        std_gui_web_set_cursor: (id, c) => {
            const w = wins.get(id);
            if (w && w.canvas.style) w.canvas.style.cursor = CURSORS[c] || "default";
        },
        std_gui_web_clipboard_set: (p, n) => {
            clipText = str(p, n);
            clipWritten = true;
            const nav = g.navigator;
            if (nav && nav.clipboard && nav.clipboard.writeText) {
                nav.clipboard.writeText(clipText).catch(() => {});
                return 1;
            }
            return 1;   // the page's own copy event still carries it (see 'copy' above)
        },
        std_gui_web_clipboard_get: (p, cap) => {
            if (clipText === null) return -1;
            const bytes = encoder.encode(clipText);
            const n = Math.min(bytes.length, cap);
            if (n > 0) new Uint8Array(mem(), p, n).set(bytes.subarray(0, n));
            return bytes.length;
        },
    };

    const gui = {
        imports: { env: { ...libcFallbacks(() => instance, (t) => ((g.console || console).log)(t)), ...env, ...(options.imports || {}) } },
        attach(inst) { instance = inst; },
        get instance() { return instance; },
        canvas(id) { const w = wins.get(id); return w ? w.canvas : null; },
        windows() { return Array.from(wins.values()).map((w) => ({ id: w.id, canvas: w.canvas, presents: w.presents || 0 })); },
        // One frame by hand (tests, or a host with its own loop): 1 while the app runs.
        frame(ts) { return instance.exports.std_gui_web_frame(ts ?? env.std_gui_web_now()); },
        start(entry = "start", ...args) {
            const ex = instance.exports;
            let result;
            if (entry && typeof ex[entry] === "function") result = ex[entry](...args);
            else if (entry) throw new Error(`gui_web.js: the module has no export '${entry}'`);
            if (typeof ex.std_gui_web_frame === "function" && !running) {
                running = true;
                const raf = g.requestAnimationFrame || ((cb) => g.setTimeout(() => cb(env.std_gui_web_now()), 16));
                const tick = (ts) => {
                    rafHandle = null;
                    if (!running) return;
                    if (ex.std_gui_web_frame(ts)) rafHandle = raf(tick);
                    else running = false;
                };
                tick(env.std_gui_web_now());
            }
            return result;
        },
        stop() {
            running = false;
            if (rafHandle !== null && g.cancelAnimationFrame) g.cancelAnimationFrame(rafHandle);
            rafHandle = null;
        },
        get running() { return running; },
    };
    return gui;
}

export async function loadSafeCGui(source, options = {}) {
    const gui = createSafeCGui(options);
    let instance;
    if (options.loadSafeC) {
        const app = await options.loadSafeC(source, { ...options, imports: gui.imports.env });
        instance = app.instance;
    } else {
        let src = await source;
        if (src instanceof WebAssembly.Module) {
            instance = await WebAssembly.instantiate(src, gui.imports);
        } else {
            if (src && typeof src.arrayBuffer === "function") src = await src.arrayBuffer();
            instance = (await WebAssembly.instantiate(src, gui.imports)).instance;
        }
    }
    gui.attach(instance);
    const entry = options.entry === undefined ? "start" : options.entry;
    if (entry !== false) gui.start(entry);
    return { instance, exports: instance.exports, gui };
}
