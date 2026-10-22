// A minimal in-memory DOM for testing std/wasm/safec_dom.js under Node:
// elements, text nodes, simple selectors (tag, #id, .class, [attr],
// [attr="v"], compounds of those, descendant combinator), events with
// bubbling, classList, style, form-control properties, Storage, location
// + history, and a scripted fetch.

class Node {
    constructor(doc) { this.ownerDocument = doc; this.parentNode = null; this.childNodes = []; this.listeners = {}; }
    get parentElement() { return this.parentNode instanceof Element ? this.parentNode : null; }
    get children() { return this.childNodes.filter((n) => n instanceof Element); }
    get firstElementChild() { return this.children[0] || null; }
    get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
    get nextElementSibling() { return this.sibling(1); }
    get previousElementSibling() { return this.sibling(-1); }
    sibling(dir) {
        if (!this.parentNode) return null;
        const c = this.parentNode.children;
        return c[c.indexOf(this) + dir] || null;
    }
    appendChild(n) { return this.insertBefore(n, null); }
    insertBefore(n, ref) {
        if (n.parentNode) n.remove();
        const i = ref ? this.childNodes.indexOf(ref) : -1;
        if (i < 0) this.childNodes.push(n); else this.childNodes.splice(i, 0, n);
        n.parentNode = this;
        return n;
    }
    remove() {
        if (!this.parentNode) return;
        const c = this.parentNode.childNodes;
        c.splice(c.indexOf(this), 1);
        this.parentNode = null;
    }
    get textContent() { return this.childNodes.map((n) => n.textContent).join(""); }
    set textContent(v) {
        for (const c of this.childNodes) c.parentNode = null;
        this.childNodes = [];
        if (v !== "") this.appendChild(this.ownerDocument.createTextNode(String(v)));
    }
    addEventListener(type, fn, opts) {
        (this.listeners[type] ||= []).push({ fn, once: !!(opts && opts.once) });
    }
    removeEventListener(type, fn) {
        this.listeners[type] = (this.listeners[type] || []).filter((l) => l.fn !== fn);
    }
    dispatchEvent(ev) {
        ev.target ||= this;
        for (let n = this; n && !ev._stopped; n = n.parentNode || (n === n.ownerDocument ? n.defaultView : null)) {
            ev.currentTarget = n;
            for (const l of [...(n.listeners[ev.type] || [])]) {
                if (l.once) n.removeEventListener(ev.type, l.fn);
                l.fn(ev);
            }
            if (n === n.ownerDocument?.defaultView) break;
        }
        return !ev.defaultPrevented;
    }
    querySelectorAll(sel) {
        const out = [];
        const walk = (n) => {
            for (const c of n.children) { if (matches(c, sel, this)) out.push(c); walk(c); }
        };
        walk(this);
        return out;
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

class Text extends Node {
    constructor(doc, data) { super(doc); this.data = data; }
    get textContent() { return this.data; }
    set textContent(v) { this.data = String(v); }
}

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

class Element extends Node {
    constructor(doc, tag) {
        super(doc);
        this.tagName = tag.toUpperCase();
        this.attrs = new Map();
        const self = this;
        this.classList = {
            list() { return (self.attrs.get("class") || "").split(/\s+/).filter(Boolean); },
            set(l) { self.attrs.set("class", l.join(" ")); },
            add(c) { const l = this.list(); if (!l.includes(c)) { l.push(c); this.set(l); } },
            remove(c) { this.set(this.list().filter((x) => x !== c)); },
            contains(c) { return this.list().includes(c); },
            toggle(c) { if (this.contains(c)) { this.remove(c); return false; } this.add(c); return true; },
        };
        const props = new Map();
        this.style = {
            setProperty(k, v) { props.set(k, v); },
            removeProperty(k) { props.delete(k); },
            getPropertyValue(k) { return props.get(k) || ""; },
            get display() { return props.get("display") || ""; },
        };
        this.value = tag === "input" || tag === "textarea" || tag === "select" ? "" : undefined;
        this.checked = false;
        this.disabled = false;
        this.focused = false;
    }
    get id() { return this.attrs.get("id") || ""; }
    setAttribute(k, v) { this.attrs.set(k, String(v)); }
    getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
    removeAttribute(k) { this.attrs.delete(k); }
    get innerHTML() {
        return this.childNodes.map((n) => n instanceof Text ? esc(n.data) : n.outerHTML).join("");
    }
    set innerHTML(html) {
        // Enough of a parser for tests: text, and <tag attr="v">...</tag>.
        this.textContent = "";
        const stack = [this];
        const re = /<\/([a-z0-9]+)>|<([a-z0-9]+)((?:\s+[a-z-]+="[^"]*")*)\s*>|([^<]+)/gi;
        let m;
        while ((m = re.exec(html))) {
            const top = stack[stack.length - 1];
            if (m[1]) stack.pop();
            else if (m[2]) {
                const el = this.ownerDocument.createElement(m[2].toLowerCase());
                for (const a of m[3].matchAll(/([a-z-]+)="([^"]*)"/gi)) el.setAttribute(a[1], a[2]);
                top.appendChild(el);
                stack.push(el);
            } else top.appendChild(this.ownerDocument.createTextNode(m[4]));
        }
    }
    get outerHTML() {
        const attrs = [...this.attrs].map(([k, v]) => ` ${k}="${v}"`).join("");
        const t = this.tagName.toLowerCase();
        return `<${t}${attrs}>${this.innerHTML}</${t}>`;
    }
    focus() { this.focused = true; this.ownerDocument.activeElement = this; }
    blur() { this.focused = false; }
    click() { this.dispatchEvent(new Event("click", { bubbles: true })); }
}

function matchesCompound(el, c) {
    const re = /([a-z0-9]+)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/gi;
    let m;
    let any = false;
    while ((m = re.exec(c))) {
        any = true;
        if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
        if (m[2] && el.id !== m[2]) return false;
        if (m[3] && !el.classList.contains(m[3])) return false;
        if (m[4] && (m[5] === undefined ? !el.attrs.has(m[4]) : el.getAttribute(m[4]) !== m[5])) return false;
    }
    return any;
}

function matches(el, sel, root) {
    return sel.split(",").some((alt) => {
        const parts = alt.trim().split(/\s+/);
        if (!matchesCompound(el, parts[parts.length - 1])) return false;
        let n = el.parentNode;
        for (let i = parts.length - 2; i >= 0; i--) {
            while (n && n !== root && !(n instanceof Element && matchesCompound(n, parts[i]))) n = n.parentNode;
            if (!n || n === root) return false;
            n = n.parentNode;
        }
        return true;
    });
}

export class Event {
    constructor(type, init = {}) { Object.assign(this, init); this.type = type; this.defaultPrevented = false; this.timeStamp = 1; }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() { this._stopped = true; }
}

class Storage {
    constructor() { this.m = new Map(); }
    get length() { return this.m.size; }
    key(i) { return [...this.m.keys()][i] ?? null; }
    getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
    setItem(k, v) { if (String(v).length > 1000) throw new Error("QuotaExceededError"); this.m.set(k, String(v)); }
    removeItem(k) { this.m.delete(k); }
    clear() { this.m.clear(); }
}

export function createFakeWindow() {
    const document = new Node(null);
    Object.setPrototypeOf(document, Node.prototype);
    document.ownerDocument = document;
    document.createElement = (t) => new Element(document, t);
    document.createTextNode = (s) => new Text(document, s);
    const html = document.appendChild(document.createElement("html"));
    document.head = html.appendChild(document.createElement("head"));
    document.body = html.appendChild(document.createElement("body"));
    document.getElementById = (id) => document.querySelector("#" + id);
    document.title = "";

    const location = {
        origin: "http://example.test", host: "example.test",
        pathname: "/", search: "", hash: "",
        get href() { return this.origin + this.pathname + this.search + this.hash; },
        set(url) {
            const u = new URL(url, this.href);
            this.pathname = u.pathname; this.search = u.search; this.hash = u.hash;
        },
        assign(url) { this.set(url); this.assigned = url; },
    };
    const stack = [{ url: "/", state: null }];
    let pos = 0;
    const window = {
        document, location,
        localStorage: new Storage(), sessionStorage: new Storage(),
        listeners: {},
        addEventListener: Node.prototype.addEventListener,
        removeEventListener: Node.prototype.removeEventListener,
        dispatchEvent(ev) { ev.target ||= window; ev.currentTarget = window; for (const l of [...(this.listeners[ev.type] || [])]) l.fn(ev); return true; },
        history: {
            pushState(state, _t, url) { stack.length = pos + 1; stack.push({ url, state }); pos++; location.set(url); },
            replaceState(state, _t, url) { stack[pos] = { url, state }; location.set(url); },
            back() { if (pos > 0) { pos--; location.set(stack[pos].url); window.dispatchEvent(new Event("popstate", { state: stack[pos].state })); } },
            forward() { if (pos + 1 < stack.length) { pos++; location.set(stack[pos].url); window.dispatchEvent(new Event("popstate", { state: stack[pos].state })); } },
            get length() { return stack.length; },
        },
        routes: new Map(),
        async fetch(url, init = {}) {
            const r = window.routes.get(url);
            if (!r) throw new Error("network error: " + url);
            const body = typeof r === "function" ? r(init) : r;
            return {
                status: body.status ?? 200, statusText: body.statusText ?? "OK", url: "http://example.test" + url,
                headers: { get: (k) => (body.headers || {})[k.toLowerCase()] ?? null },
                text: async () => body.text ?? "",
            };
        },
        setTimeout, clearTimeout, setInterval, clearInterval,
        requestAnimationFrame: (cb) => setTimeout(() => cb(16.5), 1),
        cancelAnimationFrame: (h) => clearTimeout(h),
        performance: { now: () => 1234.5 },
    };
    document.defaultView = window;
    return window;
}
