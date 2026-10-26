// SafeC Standard Library — JavaScript glue for std/gui on WebAssembly
// (gui_web.h, gui_web_wasm.sc).
//
// Implements the env.std_gui_web_* imports a wasm32 module built with
// std/gui needs — each GuiWindow a <canvas> (in the page's flow, or in a
// movable, resizable frame), input forwarded to the module (keyboard and
// IME text through a hidden <textarea> that holds the real DOM focus),
// presentation with putImageData or, on request, WebGL2/WebGL1 — and drives
// the module's GuiApp from requestAnimationFrame. An ES module with no
// dependencies; works in browsers and, with stand-ins passed through
// `options.globals`, in Node.
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
//   windows  "canvas" (default): each window is its canvas, in the page's
//            flow. "frames": each window is a <div class="safec-gui-window">
//            positioned in 'container' (default <body>) with a title bar
//            (drag to move: GUI_EVENT_MOVE; × : GUI_EVENT_CLOSE) and a
//            resize handle (GUI_EVENT_RESIZE), raised on a click —
//            undecorated windows get neither bar nor handle. Style them
//            with CSS (.safec-gui-window, .safec-gui-titlebar,
//            .safec-gui-close, .safec-gui-resize).
//   container  frames mode: the element (or selector) the frames go in.
//   present  "webgl": windows present through WebGL2 (else WebGL1) — the
//            premultiplied frame uploaded as a texture, the damaged
//            rectangle only, drawn as one quad; any failure or a lost
//            context falls back to putImageData. "2d": putImageData always.
//            Default: putImageData unless a window asks for
//            GUI_PRESENT_GPU (gui_gpu.h).
//   preserveDrawingBuffer  WebGL: keep frames readable (toDataURL, tests).
//   domGlue  std/wasm/safec_dom.js's glue object (loadSafeCGui sets it when
//            given loadSafeC): gui_web_window_element()/gui_web_mount()
//            exchange elements with dom.h through its handles.
//   entry    loadSafeCGui: the export to call once instantiated (default
//            "start"; false: none — call gui.start() yourself).
//   loadSafeC  loadSafeCGui: std/wasm/safec_dom.js's loadSafeC, to
//            instantiate through it (DOM imports + these).
//   files    { "/path": Uint8Array | ArrayBuffer }: read-only files the
//            module can fopen()/fread() (e.g. the fonts std/gui looks for
//            in /usr/share/fonts/...; see fetchFiles below).
//   imports  extra env imports (yours win).
//   globals  where to find document, window, navigator, performance,
//            requestAnimationFrame, ImageData, ResizeObserver, matchMedia,
//            OffscreenCanvas, setTimeout (default: globalThis).
// gui.start(entry) returns the entry's result; gui.stop() ends the frame
// loop; gui.canvas(id) is a window's canvas (id = GuiWindow.id).

const EV = { CLOSE: 1, MOVE: 12, MOUSE_DOWN: 3, MOUSE_UP: 4, MOUSE_MOVE: 5, KEY_DOWN: 6, KEY_UP: 7, CHAR: 8, SCROLL: 9,
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
// arguments, each int/pointer 4 bytes, long/long long/size_t/double 8
// (8-aligned) — SafeC's long and unsigned long are 64-bit on every target.
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
            const wide = len === "ll" || len === "l" || len === "j" || len === "z" || len === "t";
            let v = wide ? take8i() : BigInt(take4());
            const signed = conv === "d" || conv === "i";
            if (!signed) v = BigInt.asUintN(wide ? 64 : 32, v);
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
function libcFallbacks(getInstance, log, files) {
    const view = () => new DataView(getInstance().exports.memory.buffer);
    const cstr = (p) => { const u8 = new Uint8Array(getInstance().exports.memory.buffer); let e = p; while (u8[e] !== 0) e++; return new TextDecoder().decode(u8.subarray(p, e)); };
    const encoder = new TextEncoder();
    const snprintf = (buf, n, fmt, va) => {
        n = Number(n);   // (size_t: i64 in SafeC declarations)
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
    const open = [];
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
        // read-only files from options.files (fonts, images): FILE* is an
        // index into 'open' (1-based)
        fopen: (p, m) => {
            const data = files.get(cstr(p));
            if (!data || cstr(m).includes("w") || cstr(m).includes("a")) return 0;
            const h = open.indexOf(null) >= 0 ? open.indexOf(null) : open.length;
            open[h] = { data, pos: 0, err: 0 };
            return h + 1;
        },
        fclose: (f) => { if (f > 0 && f <= open.length) open[f - 1] = null; return 0; },
        fread: (buf, size, n, f) => {
            size = Number(size); n = Number(n);
            const o = f > 0 ? open[f - 1] : null;
            if (!o || size === 0) return 0;
            const want = Math.min(size * n, o.data.length - o.pos);
            const items = Math.floor(Math.max(0, want) / size);
            new Uint8Array(getInstance().exports.memory.buffer, buf, items * size).set(o.data.subarray(o.pos, o.pos + items * size));
            o.pos += items * size;
            return items;
        },
        fwrite: () => 0,
        ferror: (f) => { const o = f > 0 ? open[f - 1] : null; return o ? o.err : 1; },
        // (long offsets are i64 — BigInt — in SafeC's declarations)
        fseek: (f, off, whence) => {
            off = Number(off);
            const o = f > 0 ? open[f - 1] : null;
            if (!o) return -1;
            const base = whence === 0 ? 0 : whence === 1 ? o.pos : o.data.length;
            if (base + off < 0) return -1;
            o.pos = base + off;
            return 0;
        },
        ftell: (f) => { const o = f > 0 ? open[f - 1] : null; return BigInt(o ? o.pos : -1); },
        feof: (f) => { const o = f > 0 ? open[f - 1] : null; return o && o.pos >= o.data.length ? 1 : 0; },
        fflush: () => 0,
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
    let domGlue = options.domGlue || null;
    // options.files: { path: Uint8Array | ArrayBuffer } read through fopen/fread
    const files = new Map(Object.entries(options.files || {}).map(([k, v]) => [k, v instanceof Uint8Array ? v : new Uint8Array(v)]));
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const mem = () => instance.exports.memory.buffer;
    const str = (p, n) => (n > 0 ? decoder.decode(new Uint8Array(mem(), p, n)) : "");
    const wins = new Map();          // id -> window record
    let created = 0;
    let topZ = 10;
    let clipText = null;             // what gui_clipboard_get_text() sees
    let clipWritten = false;         // set by clipboard_set during an event
    let heldPaste = null;            // a Ctrl/Cmd+V keydown waiting for its paste event
    let running = false;
    let rafHandle = null;
    let live = null;                 // the aria-live region (gui_web_announce)
    const frames = options.windows === "frames";
    const apple = /Mac|iPhone|iPad|iPod/.test((g.navigator && (g.navigator.platform || g.navigator.userAgent)) || "");
    const doc = g.document || null;

    const send = (w, kind, f = {}) => {
        if (!instance || !wins.has(w.id)) return;
        clipWritten = false;
        instance.exports.std_gui_web_event(w.id, kind, f.x || 0, f.y || 0, f.button || 0, f.keycode || 0,
            f.codepoint || 0, f.mods || 0, f.dx || 0, f.dy || 0, f.width || 0, f.height || 0, f.scale || 0);
    };
    const sendText = (w, text) => {
        for (const ch of text) {
            const cp = ch.codePointAt(0);
            if (cp >= 32 && cp !== 127) send(w, EV.CHAR, { codepoint: cp });
        }
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
    const css = (el, props) => { if (el && el.style) Object.assign(el.style, props); };
    const clampSize = (w, lw, lh) => {
        if (w.minW > 0) lw = Math.max(lw, w.minW);
        if (w.minH > 0) lh = Math.max(lh, w.minH);
        if (w.maxW > 0) lw = Math.min(lw, w.maxW);
        if (w.maxH > 0) lh = Math.min(lh, w.maxH);
        return [Math.max(1, Math.round(lw)), Math.max(1, Math.round(lh))];
    };

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
    const setCssSize = (w) => {
        if (!w.fit) css(w.canvas, { width: `${w.lw}px`, height: `${w.lh}px` });
    };

    const pickCanvas = (index, title) => {
        const want = options.canvas;
        if (typeof want === "function") return want(index, title);
        if (want && typeof want === "object" && index === 0) return want;
        if (typeof want === "string" && index === 0 && doc) return doc.querySelector(want);
        if (doc) {
            if (!frames) {
                const free = Array.from(doc.querySelectorAll("canvas[data-safec-gui]") || []).find((c) => !c.__safecGui);
                if (free) return free;
            }
            return doc.createElement("canvas");
        }
        if (g.OffscreenCanvas) return new g.OffscreenCanvas(1, 1);
        return null;
    };
    const containerOf = () => {
        const c = options.container;
        if (c && typeof c === "object") return c;
        if (typeof c === "string" && doc) return doc.querySelector(c);
        return doc ? (doc.body || doc.documentElement) : null;
    };

    // ── canvas events ──
    const listenCanvas = (w) => {
        const c = w.canvas;
        if (!c.addEventListener) return;
        const on = (t, f, o) => { c.addEventListener(t, f, o); w.canvasOff.push(() => c.removeEventListener(t, f, o)); };
        on("pointerdown", (e) => {
            raise(w);
            focusInput(w);
            if (c.setPointerCapture && e.pointerId !== undefined) { try { c.setPointerCapture(e.pointerId); } catch (_) {} }
            send(w, EV.MOUSE_DOWN, { ...posOf(w, e), button: [0, 2, 1][e.button] ?? 0, mods: modsOf(e) });
            if (e.preventDefault) e.preventDefault();   // the hidden input keeps the focus
        });
        on("pointerup", (e) => send(w, EV.MOUSE_UP, { ...posOf(w, e), button: [0, 2, 1][e.button] ?? 0, mods: modsOf(e) }));
        on("pointermove", (e) => send(w, EV.MOUSE_MOVE, { ...posOf(w, e), mods: modsOf(e) }));
        on("pointerenter", (e) => send(w, EV.MOUSE_ENTER, posOf(w, e)));
        on("pointerleave", (e) => send(w, EV.MOUSE_LEAVE, posOf(w, e)));
        on("wheel", (e) => {
            const unit = e.deltaMode === 1 ? 1 / 3 : e.deltaMode === 2 ? 3 : 1 / 100;   // lines per unit
            send(w, EV.SCROLL, { ...posOf(w, e), dx: (e.deltaX || 0) * unit, dy: -(e.deltaY || 0) * unit, mods: modsOf(e) });
            if (e.preventDefault) e.preventDefault();
        }, { passive: false });
        on("contextmenu", (e) => { if (e.preventDefault) e.preventDefault(); });
        if (w.gl) {
            on("webglcontextlost", (e) => { w.glLost = true; if (e.preventDefault) e.preventDefault(); });
        }
    };

    // ── keyboard and text: the hidden <textarea> ──
    const focusInput = (w) => {
        if (!w.input) return;
        // over the window, so an IME's candidate window opens next to it
        const r = w.canvas.getBoundingClientRect ? w.canvas.getBoundingClientRect() : { left: 0, top: 0 };
        css(w.input, { left: `${Math.round(r.left)}px`, top: `${Math.round(r.top)}px` });
        if (doc && doc.activeElement !== w.input && w.input.focus) w.input.focus({ preventScroll: true });
    };
    const makeInput = (w) => {
        if (!doc || !doc.createElement) return;
        const t = doc.createElement("textarea");
        for (const [k, v] of [["data-safec-input", ""], ["aria-label", w.title], ["autocapitalize", "off"],
                              ["autocomplete", "off"], ["autocorrect", "off"], ["spellcheck", "false"],
                              ["tabindex", "0"], ["rows", "1"]]) t.setAttribute(k, v);
        css(t, { position: "fixed", left: "0px", top: "0px", width: "1px", height: "1px", opacity: "0",
                 border: "0", padding: "0", margin: "0", resize: "none", overflow: "hidden",
                 pointerEvents: "none", fontSize: "16px", zIndex: "-1", caretColor: "transparent" });
        (doc.body || doc.documentElement).appendChild(t);
        w.input = t;
        const on = (type, f) => { t.addEventListener(type, f); w.off.push(() => t.removeEventListener(type, f)); };
        let composing = false;
        on("focus", () => { if (!w.focused) { w.focused = true; send(w, EV.FOCUS_IN); } });
        on("blur", () => { if (w.focused) { w.focused = false; send(w, EV.FOCUS_OUT); } });
        on("compositionstart", () => { composing = true; });
        on("compositionend", (e) => {
            composing = false;
            const text = e.data ?? t.value;
            t.value = "";
            if (text) sendText(w, text);
        });
        on("input", (e) => {
            // IME commits, mobile keyboards, context-menu pastes: text with
            // no keydown that carried it
            if (composing || e.isComposing) return;
            const text = t.value;
            t.value = "";
            if (text) sendText(w, text);
        });
        on("keydown", (e) => {
            if (composing || e.isComposing || e.keyCode === 229 || e.key === "Process") return;   // the IME's
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
                e.preventDefault();   // (typed text arrives through KEY_DOWN, not 'input')
            }
        });
        on("keyup", (e) => {
            if (e.isComposing) return;
            send(w, EV.KEY_UP, { keycode: keyCodeOf(e), mods: modsOf(e) });
        });
    };

    const docListeners = () => {
        if (!doc || !doc.addEventListener || docListeners.done) return;
        docListeners.done = true;
        doc.addEventListener("paste", (e) => {
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
        doc.addEventListener("copy", copy);
        doc.addEventListener("cut", copy);
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

    // ── frames ──
    const raise = (w) => { if (w.frame) css(w.frame, { zIndex: String(++topZ) }); };
    const drag = (w, handle, onMove) => {
        let start = null;
        const down = (e) => {
            start = { x: e.clientX, y: e.clientY, lx: w.x, ly: w.y, lw: w.lw, lh: w.lh };
            raise(w);
            if (handle.setPointerCapture && e.pointerId !== undefined) { try { handle.setPointerCapture(e.pointerId); } catch (_) {} }
            if (e.preventDefault) e.preventDefault();
            if (e.stopPropagation) e.stopPropagation();
        };
        const move = (e) => { if (start) onMove(e.clientX - start.x, e.clientY - start.y, start); };
        const up = () => { start = null; };
        handle.addEventListener("pointerdown", down);
        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", up);
        w.off.push(() => { handle.removeEventListener("pointerdown", down); handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); });
    };
    const makeFrame = (w, decorated, resizable) => {
        const f = doc.createElement("div");
        f.setAttribute("class", "safec-gui-window");
        f.setAttribute("role", "dialog");
        f.setAttribute("aria-label", w.title);
        css(f, { position: "absolute", left: `${w.x}px`, top: `${w.y}px`, zIndex: String(++topZ),
                 display: "flex", flexDirection: "column", background: "#fff", borderRadius: "8px",
                 boxShadow: "0 4px 24px rgba(0,0,0,.25)", overflow: "hidden" });
        if (decorated) {
            const bar = doc.createElement("div");
            bar.setAttribute("class", "safec-gui-titlebar");
            css(bar, { display: "flex", alignItems: "center", height: "28px", padding: "0 4px 0 10px",
                       font: "13px system-ui, sans-serif", background: "#e9ecef", cursor: "default", userSelect: "none" });
            const title = doc.createElement("span");
            title.textContent = w.title;
            css(title, { flex: "1", overflow: "hidden", whiteSpace: "nowrap", textOverflow: "ellipsis" });
            const close = doc.createElement("button");
            close.setAttribute("class", "safec-gui-close");
            close.setAttribute("aria-label", "Close");
            close.textContent = "×";
            css(close, { border: "0", background: "transparent", font: "18px system-ui, sans-serif", width: "24px", height: "24px", cursor: "pointer" });
            close.addEventListener("click", () => send(w, EV.CLOSE));
            bar.appendChild(title);
            bar.appendChild(close);
            f.appendChild(bar);
            w.bar = bar;
            w.barTitle = title;
            drag(w, bar, (dx, dy, s) => {
                w.x = s.lx + dx;
                w.y = s.ly + dy;
                css(f, { left: `${w.x}px`, top: `${w.y}px` });
                send(w, EV.MOVE, { x: w.x, y: w.y });
            });
        }
        f.appendChild(w.canvas);
        if (decorated && resizable) {
            const grip = doc.createElement("div");
            grip.setAttribute("class", "safec-gui-resize");
            css(grip, { position: "absolute", right: "0px", bottom: "0px", width: "14px", height: "14px", cursor: "nwse-resize" });
            f.appendChild(grip);
            drag(w, grip, (dx, dy, s) => {
                const [lw, lh] = clampSize(w, s.lw + dx, s.lh + dy);
                if (lw === w.lw && lh === w.lh) return;
                w.lw = lw;
                w.lh = lh;
                setCssSize(w);
                relayout(w, EV.RESIZE);
            });
        }
        f.addEventListener("pointerdown", () => raise(w));
        w.frame = f;
        containerOf().appendChild(f);
    };

    // ── 2D and WebGL presentation ──
    const ctx2d = (w) => (w.ctx2d ||= w.canvas.getContext("2d"));
    // A fresh canvas in the old one's place (a canvas keeps the kind of the
    // first context it gave out: switching between WebGL and 2D needs a new one).
    const replaceCanvas = (w) => {
        const old = w.canvas;
        const c = doc ? doc.createElement("canvas") : new g.OffscreenCanvas(old.width, old.height);
        c.width = old.width;
        c.height = old.height;
        if (old.style && c.style) for (const k of ["width", "height", "display", "cursor", "touchAction"]) if (old.style[k] !== undefined) c.style[k] = old.style[k];
        if (old.getAttribute && c.setAttribute) for (const k of ["class", "id", "role", "aria-label", "data-safec-gui", "data-safec-fit"]) { const v = old.getAttribute(k); if (v !== null) c.setAttribute(k, v); }
        if (old.parentNode) { old.parentNode.insertBefore(c, old); old.remove(); }
        for (const f of w.canvasOff) f();
        w.canvasOff = [];
        old.__safecGui = false;
        c.__safecGui = true;
        if (w.ro) { w.ro.disconnect(); w.ro.observe(c); }
        if (w.el === old) w.el = c;
        w.canvas = c;
        w.ctx2d = null;
        w.image = null;
        listenCanvas(w);
    };
    const GL_VS = "attribute vec2 p; varying vec2 uv; void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }";
    const GL_FS = "precision mediump float; varying vec2 uv; uniform sampler2D t; void main() { gl_FragColor = texture2D(t, uv); }";
    const glAttach = (w) => {
        if (w.ctx2d) replaceCanvas(w);
        const attrs = { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
                        preserveDrawingBuffer: !!options.preserveDrawingBuffer };
        let gl = null, v2 = false;
        try {
            gl = w.canvas.getContext("webgl2", attrs);
            v2 = !!gl;
            if (!gl) gl = w.canvas.getContext("webgl", attrs) || w.canvas.getContext("experimental-webgl", attrs);
        } catch (_) { gl = null; }
        if (!gl) return 0;
        const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null; };
        const vs = sh(gl.VERTEX_SHADER, GL_VS), fs = sh(gl.FRAGMENT_SHADER, GL_FS);
        if (!vs || !fs) { replaceCanvas(w); return 0; }
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.bindAttribLocation(prog, 0, "p");
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { replaceCanvas(w); return 0; }
        gl.useProgram(prog);
        gl.uniform1i(gl.getUniformLocation(prog, "t"), 0);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        const tex = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                              [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.disable(gl.BLEND);
        if (gl.getError() !== gl.NO_ERROR) { replaceCanvas(w); return 0; }
        w.gl = { gl, v2, prog, buf, tex, tw: 0, th: 0 };
        w.glLost = false;
        // (re)listen: the context-lost handler
        for (const f of w.canvasOff) f();
        w.canvasOff = [];
        listenCanvas(w);
        return 1;
    };
    const glPresent = (w, ptr, bw, bh, x, y, rw, rh) => {
        const s = w.gl;
        if (!s || w.glLost || (s.gl.isContextLost && s.gl.isContextLost())) return 0;
        const gl = s.gl;
        const all = new Uint8Array(mem(), ptr, bw * bh * 4);
        gl.bindTexture(gl.TEXTURE_2D, s.tex);
        if (s.tw !== bw || s.th !== bh) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, bw, bh, 0, gl.RGBA, gl.UNSIGNED_BYTE, all);
            s.tw = bw;
            s.th = bh;
        } else if (s.v2) {
            // just the damaged rectangle, straight out of linear memory
            gl.pixelStorei(gl.UNPACK_ROW_LENGTH, bw);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, rw, rh, gl.RGBA, gl.UNSIGNED_BYTE, all, (y * bw + x) * 4);
            gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
        } else {
            // WebGL1 has no row length: the damaged rows, full width
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, y, bw, rh, gl.RGBA, gl.UNSIGNED_BYTE, all.subarray(y * bw * 4, (y + rh) * bw * 4));
        }
        const cw = w.canvas.width, ch = w.canvas.height;
        if (cw !== bw || ch !== bh) {
            gl.viewport(0, 0, cw, ch);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
        gl.viewport(0, ch - bh, bw, bh);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        w.presents = (w.presents || 0) + 1;
        return gl.getError() === gl.CONTEXT_LOST_WEBGL ? 0 : 1;
    };
    const glDetach = (w) => {
        const s = w.gl;
        if (!s) return;
        if (!w.glLost) { try { s.gl.deleteTexture(s.tex); s.gl.deleteBuffer(s.buf); s.gl.deleteProgram(s.prog); } catch (_) {} }
        w.gl = null;
        replaceCanvas(w);
    };

    const announce = (text) => {
        if (!doc || !doc.createElement) return;
        if (!live) {
            live = doc.createElement("div");
            live.setAttribute("role", "status");
            live.setAttribute("aria-live", "polite");
            live.setAttribute("data-safec-announce", "");
            css(live, { position: "fixed", left: "-10000px", width: "1px", height: "1px", overflow: "hidden" });
            (doc.body || doc.documentElement).appendChild(live);
        }
        live.textContent = text;
    };

    const env = {
        std_gui_web_now: () => (g.performance ? g.performance.now() : Date.now()),
        std_gui_web_dpr: () => g.devicePixelRatio || 1,
        std_gui_web_primary_mod: () => (apple ? MOD.META : MOD.CTRL),
        std_gui_web_present_mode: () => (options.present === "webgl" || options.present === "gpu" ? 1 : options.present === "2d" || options.present === "cpu" ? 2 : 0),
        // opts: [width, height, hiDpi, visible, x, y, decorated, resizable, minW, minH, maxW, maxH]
        // out: [logical width, logical height, scale * 1000, x, y]
        std_gui_web_create: (id, tp, tn, optsPtr, out) => {
            const o = new Int32Array(mem(), optsPtr, 12);
            const [w0, h0, hiDpi, visible, ox, oy, decorated, resizable, minW, minH, maxW, maxH] = Array.from(o);
            const title = str(tp, tn);
            const index = created;
            const canvas = pickCanvas(index, title);
            if (!canvas) return 0;
            if (created === 0 && options.title && doc) doc.title = title;
            created++;
            canvas.__safecGui = true;
            const POS_DEFAULT = -2147483647;
            const w = { id, title, canvas, el: canvas, lw: w0, lh: h0, hiDpi: !!hiDpi, scale: 0, off: [], canvasOff: [],
                        minW, minH, maxW, maxH, x: ox === POS_DEFAULT ? 24 + 32 * index : ox, y: oy === POS_DEFAULT ? 24 + 32 * index : oy,
                        fit: !!(options.fit || (canvas.hasAttribute && canvas.hasAttribute("data-safec-fit"))) };
            w.scale = dprOf(w);
            if (w.fit && canvas.clientWidth > 0 && canvas.clientHeight > 0) {
                w.lw = canvas.clientWidth;
                w.lh = canvas.clientHeight;
            } else {
                setCssSize(w);
            }
            canvas.width = Math.max(1, Math.round(w.lw * w.scale));
            canvas.height = Math.max(1, Math.round(w.lh * w.scale));
            if (canvas.setAttribute) {
                canvas.setAttribute("aria-label", title);
                canvas.setAttribute("role", "img");
            }
            css(canvas, { touchAction: "none", display: "block" });
            if (frames && doc) {
                makeFrame(w, !!decorated, !!resizable);
                w.el = w.frame;
            } else if (doc && !canvas.parentNode && canvas.setAttribute) {
                (doc.body || doc.documentElement).appendChild(canvas);
            }
            if (!visible) css(w.el, { display: "none" });
            wins.set(id, w);
            listenCanvas(w);
            makeInput(w);
            docListeners();
            watchScale();
            if (w.fit && g.ResizeObserver) {
                w.ro = new g.ResizeObserver(() => relayout(w, EV.RESIZE));
                w.ro.observe(canvas);
            }
            const v = new Int32Array(mem(), out, 5);
            v[0] = w.lw; v[1] = w.lh; v[2] = Math.round(w.scale * 1000);
            v[3] = frames ? w.x : POS_DEFAULT; v[4] = frames ? w.y : POS_DEFAULT;
            return 1;
        },
        std_gui_web_destroy: (id) => {
            const w = wins.get(id);
            if (!w) return;
            for (const f of w.off) f();
            for (const f of w.canvasOff) f();
            if (w.ro) w.ro.disconnect();
            if (w.input && w.input.remove) w.input.remove();
            if (w.frame && w.frame.remove) w.frame.remove();
            w.canvas.__safecGui = false;
            wins.delete(id);
        },
        std_gui_web_present: (id, ptr, bw, bh, x, y, rw, rh) => {
            const w = wins.get(id);
            if (!w) return;
            if (w.gl) glDetach(w);   // (a GPU presenter gave up: back to 2D)
            const ctx = ctx2d(w);
            if (!ctx) return;
            const buf = mem();
            if (!w.image || w.imageBuf !== buf || w.imagePtr !== ptr || w.image.width !== bw || w.image.height !== bh) {
                w.image = new g.ImageData(new Uint8ClampedArray(buf, ptr, bw * bh * 4), bw, bh);
                w.imageBuf = buf;
                w.imagePtr = ptr;
            }
            ctx.putImageData(w.image, 0, 0, x, y, rw, rh);
            w.presents = (w.presents || 0) + 1;
        },
        std_gui_web_gl_attach: (id) => { const w = wins.get(id); return w ? glAttach(w) : 0; },
        std_gui_web_gl_present: (id, ptr, bw, bh, x, y, rw, rh) => { const w = wins.get(id); return w ? glPresent(w, ptr, bw, bh, x, y, rw, rh) : 0; },
        std_gui_web_gl_detach: (id) => { const w = wins.get(id); if (w) glDetach(w); },
        std_gui_web_set_title: (id, tp, tn) => {
            const w = wins.get(id);
            if (!w) return;
            const t = str(tp, tn);
            w.title = t;
            for (const el of [w.canvas, w.frame]) if (el && el.setAttribute) el.setAttribute("aria-label", t);
            if (w.input) w.input.setAttribute("aria-label", t);
            if (w.barTitle) w.barTitle.textContent = t;
            if (options.title && doc && wins.keys().next().value === id) doc.title = t;
        },
        std_gui_web_set_size: (id, lw, lh) => {
            const w = wins.get(id);
            if (!w) return;
            w.lw = lw;
            w.lh = lh;
            css(w.canvas, { width: `${lw}px`, height: `${lh}px` });
            w.canvas.width = Math.max(1, Math.round(lw * w.scale));
            w.canvas.height = Math.max(1, Math.round(lh * w.scale));
        },
        std_gui_web_set_position: (id, x, y) => {
            const w = wins.get(id);
            if (!w || !w.frame) return;
            w.x = x;
            w.y = y;
            css(w.frame, { left: `${x}px`, top: `${y}px` });
        },
        std_gui_web_update_limits: (id, minW, minH, maxW, maxH) => {
            const w = wins.get(id);
            if (w) Object.assign(w, { minW, minH, maxW, maxH });
        },
        std_gui_web_set_visible: (id, v) => { const w = wins.get(id); if (w) css(w.el, { display: v ? (w.frame ? "flex" : "block") : "none" }); },
        std_gui_web_focus: (id) => { const w = wins.get(id); if (w) { raise(w); focusInput(w); } },
        std_gui_web_set_cursor: (id, c) => { const w = wins.get(id); if (w) css(w.canvas, { cursor: CURSORS[c] || "default" }); },
        std_gui_web_clipboard_set: (p, n) => {
            clipText = str(p, n);
            clipWritten = true;
            const nav = g.navigator;
            if (nav && nav.clipboard && nav.clipboard.writeText) nav.clipboard.writeText(clipText).catch(() => {});
            return 1;   // (without the API the page's own copy event still carries it: see 'copy' above)
        },
        std_gui_web_clipboard_get: (p, cap) => {
            if (clipText === null) return -1;
            const bytes = encoder.encode(clipText);
            const n = Math.min(bytes.length, cap);
            if (n > 0) new Uint8Array(mem(), p, n).set(bytes.subarray(0, n));
            return bytes.length;
        },
        std_gui_web_say: (p, n) => announce(str(p, n)),
        std_gui_web_dom_handle: (id) => { const w = wins.get(id); return w && domGlue ? domGlue.handle(w.el) : -1; },
        std_gui_web_mount_in: (id, h) => {
            const w = wins.get(id);
            const parent = w && domGlue ? domGlue.element(h) : null;
            if (!parent || !parent.appendChild) return 0;
            parent.appendChild(w.el);
            if (w.fit) relayout(w, EV.RESIZE);
            return 1;
        },
    };

    const gui = {
        imports: { env: { ...libcFallbacks(() => instance, (t) => ((g.console || console).log)(t), files), ...env, ...(options.imports || {}) } },
        attach(inst, glue) { instance = inst; if (glue) domGlue = glue; },
        get instance() { return instance; },
        get domGlue() { return domGlue; },
        canvas(id) { const w = wins.get(id); return w ? w.canvas : null; },
        element(id) { const w = wins.get(id); return w ? w.el : null; },
        input(id) { const w = wins.get(id); return w ? w.input : null; },
        presenter(id) { const w = wins.get(id); return !w ? null : w.gl ? (w.gl.v2 ? "webgl2" : "webgl") : "2d"; },
        windows() { return Array.from(wins.values()).map((w) => ({ id: w.id, canvas: w.canvas, element: w.el, presents: w.presents || 0 })); },
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

// { path: url } -> { path: Uint8Array }, fetched in parallel (for
// options.files). Missing files are left out.
export async function fetchFiles(map, fetchFn = globalThis.fetch) {
    const out = {};
    await Promise.all(Object.entries(map).map(async ([path, url]) => {
        try {
            const r = await fetchFn(url);
            if (r.ok !== false) out[path] = new Uint8Array(await r.arrayBuffer());
        } catch (_) { /* left out */ }
    }));
    return out;
}

export async function loadSafeCGui(source, options = {}) {
    const gui = createSafeCGui(options);
    let instance, glue = null;
    if (options.loadSafeC) {
        const app = await options.loadSafeC(source, { ...options, imports: gui.imports.env });
        instance = app.instance;
        glue = app.glue || null;
    } else {
        let src = await source;
        if (src instanceof WebAssembly.Module) {
            instance = await WebAssembly.instantiate(src, gui.imports);
        } else {
            if (src && typeof src.arrayBuffer === "function") src = await src.arrayBuffer();
            instance = (await WebAssembly.instantiate(src, gui.imports)).instance;
        }
    }
    gui.attach(instance, glue);
    const entry = options.entry === undefined ? "start" : options.entry;
    if (entry !== false) gui.start(entry);
    return { instance, exports: instance.exports, gui };
}
