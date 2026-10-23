// A fake <canvas> for std/gui/gui_web.js under Node, on top of
// tests/stdlib/wasm/fake_dom.mjs: canvases with width/height and a 2D
// context whose putImageData writes into a real RGBA framebuffer
// (readPixel), a CSS box from style.width/height (getBoundingClientRect,
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

export function createFakeCanvasWindow() {
    const window = createFakeWindow();
    const { document } = window;
    const create = document.createElement;
    document.createElement = (tag) => {
        const el = create(tag);
        if (tag.toLowerCase() === "canvas") makeCanvas(el);
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
        let ctx = null;
        el.getContext = (kind) => (kind === "2d" ? (ctx ||= makeContext(el)) : null);
        el.setPointerCapture = (id) => { el.captured = id; };
        const focus = el.focus.bind(el);
        el.focus = () => {
            const prev = document.activeElement;
            if (prev === el) return;
            if (prev && prev.blur) { prev.blur(); prev.dispatchEvent(new Event("blur")); }
            focus();
            el.dispatchEvent(new Event("focus"));
        };
        return el;
    }
    window.makeCanvas = (attrs = {}) => {
        const el = document.createElement("canvas");
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        document.body.appendChild(el);
        return el;
    };
    window.ImageData = ImageData;
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
