// A fake <canvas> for std/gui/gui_web.js under Node, on top of
// tests/stdlib/wasm/fake_dom.mjs: canvases with width/height and a 2D
// context whose putImageData writes into a real RGBA framebuffer
// (readPixel), or a WebGL2/WebGL1 context whose texture uploads (with
// UNPACK_ROW_LENGTH) and drawArrays land in that framebuffer too; focus and
// blur events for every element; a CSS box from style.width/height (getBoundingClientRect,
// clientWidth/Height — or a page-given size for "fit" canvases),
// ImageData, a manually triggered ResizeObserver and matchMedia, a
// recording navigator.clipboard and a manual requestAnimationFrame.
import { createFakeWindow, Event } from "../wasm/fake_dom.mjs";

export class ImageData {
    constructor(data, width, height) {
        if (!(data instanceof Uint8ClampedArray)) throw new TypeError("ImageData: not a Uint8ClampedArray");
        if (data.length !== width * height * 4) throw new RangeError("ImageData: size mismatch");
        this.data = data; this.width = width; this.height = height;
    }
}

function makeContext(canvas) {
    return {
        canvas,
        puts: 0,
        putImageData(img, dx, dy, x = 0, y = 0, w = img.width, h = img.height) {
            this.puts++;
            this.lastRect = [x, y, w, h];
            const fb = canvas.framebuffer();
            for (let yy = y; yy < y + h; yy++) {
                for (let xx = x; xx < x + w; xx++) {
                    const tx = dx + xx, ty = dy + yy;
                    if (tx < 0 || ty < 0 || tx >= canvas.width || ty >= canvas.height) continue;
                    const s = (yy * img.width + xx) * 4, d = (ty * canvas.width + tx) * 4;
                    fb[d] = img.data[s]; fb[d + 1] = img.data[s + 1]; fb[d + 2] = img.data[s + 2]; fb[d + 3] = img.data[s + 3];
                }
            }
        },
    };
}

// Just enough WebGL for gui_web.js's presenter: the texture's pixels go to
// the canvas framebuffer on drawArrays (top row first, like the real quad).
function makeGL(canvas, v2) {
    const C = { VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4, ARRAY_BUFFER: 5, STATIC_DRAW: 6,
                FLOAT: 7, TEXTURE0: 8, TEXTURE_2D: 9, TEXTURE_MIN_FILTER: 10, TEXTURE_MAG_FILTER: 11, NEAREST: 12,
                TEXTURE_WRAP_S: 13, TEXTURE_WRAP_T: 14, CLAMP_TO_EDGE: 15, UNPACK_ALIGNMENT: 16, BLEND: 17,
                NO_ERROR: 0, RGBA: 18, UNSIGNED_BYTE: 19, UNPACK_ROW_LENGTH: 20, COLOR_BUFFER_BIT: 21,
                TRIANGLE_STRIP: 22, CONTEXT_LOST_WEBGL: 0x9242 };
    let tex = null, tw = 0, th = 0, rowLength = 0, lost = false;
    const gl = {
        ...C, uploads: [], draws: 0,
        createShader: () => ({}), shaderSource() {}, compileShader() {}, getShaderParameter: () => true,
        createProgram: () => ({}), attachShader() {}, bindAttribLocation() {}, linkProgram() {}, getProgramParameter: () => true,
        useProgram() {}, uniform1i() {}, getUniformLocation: () => ({}), createBuffer: () => ({}), bindBuffer() {}, bufferData() {},
        enableVertexAttribArray() {}, vertexAttribPointer() {}, createTexture: () => ({}), activeTexture() {}, bindTexture() {},
        texParameteri() {}, disable() {}, deleteTexture() {}, deleteBuffer() {}, deleteProgram() {},
        pixelStorei(k, v) { if (k === C.UNPACK_ROW_LENGTH) { if (!v2) throw new Error("no UNPACK_ROW_LENGTH in WebGL1"); rowLength = v; } },
        getError: () => (lost ? C.CONTEXT_LOST_WEBGL : 0),
        isContextLost: () => lost,
        loseContext() { lost = true; canvas.dispatchEvent(new Event("webglcontextlost")); },
        texImage2D(t, l, ifmt, w, h, b, f, ty, data) { tw = w; th = h; tex = new Uint8Array(data); gl.uploads.push(["full", 0, 0, w, h]); },
        texSubImage2D(t, l, x, y, w, h, f, ty, data, off = 0) {
            const stride = (rowLength || w) * 4;
            for (let r = 0; r < h; r++) tex.set(data.subarray(off + r * stride, off + r * stride + w * 4), ((y + r) * tw + x) * 4);
            gl.uploads.push(["sub", x, y, w, h]);
        },
        viewport() {}, clearColor() {}, clear() {},
        drawArrays() {
            gl.draws++;
            const fb = canvas.framebuffer();
            for (let y = 0; y < Math.min(th, canvas.height); y++)
                fb.set(tex.subarray(y * tw * 4, y * tw * 4 + Math.min(tw, canvas.width) * 4), y * canvas.width * 4);
        },
    };
    return gl;
}

export function createFakeCanvasWindow() {
    const window = createFakeWindow();
    const { document } = window;
    const create = document.createElement;
    document.createElement = (tag) => {
        const el = create(tag);
        if (tag.toLowerCase() === "canvas") makeCanvas(el);
        const focus = el.focus.bind(el);
        el.focus = () => {
            const prev = document.activeElement;
            if (prev === el) return;
            if (prev && prev.blur) { prev.blur(); prev.dispatchEvent(new Event("blur")); }
            focus();
            el.dispatchEvent(new Event("focus"));
        };
        el.hasAttribute = (k) => el.attrs.has(k);
        // style.display: settable too (fake_dom's is read-only, from setProperty)
        Object.defineProperty(el.style, "display", {
            get() { return this.getPropertyValue("display"); },
            set(v) { this.setProperty("display", v); },
            configurable: true,
        });
        return el;
    };
    function makeCanvas(el) {
        let w = 300, h = 150, fb = new Uint8ClampedArray(w * h * 4);
        Object.defineProperty(el, "width", { get: () => w, set: (v) => { w = v >>> 0; fb = new Uint8ClampedArray(w * h * 4); } });
        Object.defineProperty(el, "height", { get: () => h, set: (v) => { h = v >>> 0; fb = new Uint8ClampedArray(w * h * 4); } });
        el.framebuffer = () => fb;
        el.readPixel = (x, y) => Array.from(fb.subarray((y * w + x) * 4, (y * w + x) * 4 + 4));
        el.cssBox = null;   // [w, h] given by the page (fit canvases)
        const cssSize = () => el.cssBox || [parseFloat(el.style.width) || w, parseFloat(el.style.height) || h];
        Object.defineProperty(el, "clientWidth", { get: () => cssSize()[0] });
        Object.defineProperty(el, "clientHeight", { get: () => cssSize()[1] });
        el.getBoundingClientRect = () => { const [cw, ch] = cssSize(); return { left: 10, top: 20, width: cw, height: ch }; };
        el.hasAttribute = (k) => el.attrs.has(k);
        let ctx = null, kindOf = null;
        el.getContext = (kind) => {
            if (kindOf && kindOf !== kind) return null;   // a canvas keeps its first context's kind
            if (kind === "2d") { kindOf = kind; return (ctx ||= makeContext(el)); }
            if ((kind === "webgl2" && window.webgl >= 2) || (kind === "webgl" && window.webgl >= 1)) {
                kindOf = kind;
                return (ctx ||= makeGL(el, kind === "webgl2"));
            }
            return null;
        };
        el.contextKind = () => kindOf;
        el.setPointerCapture = (id) => { el.captured = id; };
        return el;
    }
    window.makeCanvas = (attrs = {}) => {
        const el = document.createElement("canvas");
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        document.body.appendChild(el);
        return el;
    };
    window.ImageData = ImageData;
    window.webgl = 0;   // 2: WebGL2 contexts, 1: WebGL1 only, 0: none
    window.devicePixelRatio = 2;
    // ResizeObserver: fired by hand (window.resizeObservers.forEach(o => o.fire()))
    window.resizeObservers = [];
    window.ResizeObserver = class {
        constructor(cb) { this.cb = cb; this.targets = []; window.resizeObservers.push(this); }
        observe(t) { this.targets.push(t); }
        disconnect() { this.targets = []; }
        fire() { if (this.targets.length) this.cb(this.targets.map((target) => ({ target }))); }
    };
    // matchMedia: window.fireMediaChange() after changing devicePixelRatio
    window.mediaListeners = [];
    window.matchMedia = (q) => ({
        media: q,
        addEventListener: (type, fn) => window.mediaListeners.push(fn),
    });
    window.fireMediaChange = () => { const l = window.mediaListeners; window.mediaListeners = []; l.forEach((f) => f({})); };
    window.clipboardWrites = [];
    window.navigator = {
        platform: "Linux x86_64", userAgent: "node",
        clipboard: { writeText: async (t) => { window.clipboardWrites.push(t); } },
    };
    // frames by hand: window.runFrame(ts)
    let pending = null;
    window.requestAnimationFrame = (cb) => { pending = cb; return 1; };
    window.cancelAnimationFrame = () => { pending = null; };
    window.runFrame = (ts) => { const cb = pending; pending = null; if (cb) cb(ts); return !!cb; };
    window.hasFrame = () => pending !== null;
    let now = 1000;
    window.performance = { now: () => now };
    window.advance = (ms) => { now += ms; };
    return window;
}
