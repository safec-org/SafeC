// SafeC Standard Library — wasm32 DOM interop implementation (see dom.h).
#pragma once
#include <std/wasm/dom.h>

namespace std {

struct DomCallbackSlot {
    DomEventHandler handler;
    void* userdata;
    int used;
};

static struct DomCallbackSlot gDomCallbacks[DOM_MAX_CALLBACKS];
static int gDomCallbackCount = 0;

unsigned long wasm_strlen(const char* s) {
    unsigned long n = 0UL;
    unsafe {
        while (s[n] != (char)0) {
            n = n + 1UL;
        }
    }
    return n;
}

int dom_get_element(const char* selector) {
    unsigned long len = wasm_strlen(selector);
    int handle;
    unsafe { handle = js_get_element(selector, len); }
    return handle;
}

void dom_set_text(int handle, const char* text) {
    unsigned long len = wasm_strlen(text);
    unsafe { js_set_text(handle, text, len); }
}

void dom_set_attr(int handle, const char* name, const char* value) {
    unsigned long nameLen = wasm_strlen(name);
    unsigned long valLen = wasm_strlen(value);
    unsafe { js_set_attr(handle, name, nameLen, value, valLen); }
}

void dom_console_log(const char* text) {
    unsigned long len = wasm_strlen(text);
    unsafe { js_console_log(text, len); }
}

int dom_register_callback(DomEventHandler handler, void* userdata) {
    if (gDomCallbackCount >= DOM_MAX_CALLBACKS) {
        return -1;
    }
    int id = gDomCallbackCount;
    unsafe {
        gDomCallbacks[id].handler = handler;
        gDomCallbacks[id].userdata = userdata;
        gDomCallbacks[id].used = 1;
    }
    gDomCallbackCount = gDomCallbackCount + 1;
    return id;
}

int dom_on_click(const char* selector, DomEventHandler handler, void* userdata) {
    int handle = dom_get_element(selector);
    if (handle < 0) {
        return -1;
    }
    int id = dom_register_callback(handler, userdata);
    unsafe { js_add_click_listener(handle, id); }
    return handle;
}

void dom_dispatch_event(int callbackId) {
    if (callbackId < 0 || callbackId >= gDomCallbackCount) {
        return;
    }
    int used;
    unsafe { used = gDomCallbacks[callbackId].used; }
    if (!used) {
        return;
    }
    DomEventHandler h;
    void* ud;
    unsafe {
        h = gDomCallbacks[callbackId].handler;
        ud = gDomCallbacks[callbackId].userdata;
    }
    unsafe { h(ud); }
}

void wasm_itoa(long long v, char* buf, unsigned long cap) {
    unsafe {
        if (cap == 0UL) {
            return;
        }
        if (v == 0LL) {
            buf[0] = '0';
            if (cap > 1UL) { buf[1] = (char)0; } else { buf[0] = (char)0; }
            return;
        }
        int neg = 0;
        unsigned long long uv;
        if (v < 0LL) {
            neg = 1;
            uv = (unsigned long long)(-v);
        } else {
            uv = (unsigned long long)v;
        }
        char tmp[24];
        unsigned long n = 0UL;
        while (uv > 0ULL && n < 23UL) {
            tmp[n] = (char)('0' + (int)(uv % 10ULL));
            uv = uv / 10ULL;
            n = n + 1UL;
        }
        unsigned long outLen = n + (unsigned long)neg;
        if (outLen >= cap) {
            outLen = cap - 1UL;
        }
        unsigned long w = 0UL;
        if (neg && w < outLen) {
            buf[w] = '-';
            w = w + 1UL;
        }
        while (w < outLen) {
            n = n - 1UL;
            buf[w] = tmp[n];
            w = w + 1UL;
        }
        buf[w] = (char)0;
    }
}

} // namespace std
