// SafeC Standard Library — WASM client hydration implementation (see
// hydrate.h).
#pragma once
#include <std/wasm/hydrate.h>
#include <std/wasm/dom.h>
#include <std/wasm/wasm_rt.h>
#include <std/reactive/signal.h>
#include <std/mem.h>

namespace std {

struct HydrateTextBinding {
    &Signal<int> signal;
    int          elementHandle;
};

static void __hydrate_text_int_listener(?&HydrateTextBinding bq) {
    match (bq) {
        case null: return;
        case some(b): {
            char buf[24];
            unsafe {
                wasm_itoa((long long)b.signal.get(), buf, 24UL);
                dom_set_text(b.elementHandle, buf);
            }
        }
    }
}

void hydrate_bind_text_int_handle(&Signal<int> signal, int elementHandle) {
    ?&HydrateTextBinding bq = null;
    unsafe { bq = (?&HydrateTextBinding)alloc_raw_(sizeof(struct HydrateTextBinding)); }
    match (bq) {
        case null: return;
        case some(b): {
            b.signal = signal;
            b.elementHandle = elementHandle;
        }
    }
    signal_subscribe(signal, __hydrate_text_int_listener, bq);
}

// One record type for every String/int binding below; 'mode' picks what
// it writes.
#define HYD_TEXT_    0
#define HYD_ATTR_    1
#define HYD_CLASS_   2
#define HYD_VISIBLE_ 3
#define HYD_VALUE_   4
#define HYD_CHECKED_ 5

struct HydrateStrBinding {
    &Signal<String> signal;
    int             handle;
    int             mode;
    struct String   name;     // attribute name
    int             writing;  // set while the binding itself stores into the signal
};

struct HydrateIntBinding {
    &Signal<int>  signal;
    int           handle;
    int           mode;
    struct String name;       // class name
    int           writing;
};

static void hydrate_str_apply_(?&HydrateStrBinding bq) {
    match (bq) {
        case null: { }
        case some(b): {
            if (b.writing) { return; }
            &String v = b.signal.get_ref();
            if (b.mode == HYD_TEXT_) { dom_set_text(b.handle, v.as_ptr()); }
            else if (b.mode == HYD_ATTR_) { dom_set_attr(b.handle, b.name.as_ptr(), v.as_ptr()); }
            else if (b.mode == HYD_VALUE_) { dom_set_value(b.handle, v.as_ptr()); }
        }
    }
}

static void hydrate_int_apply_(?&HydrateIntBinding bq) {
    match (bq) {
        case null: { }
        case some(b): {
            if (b.writing) { return; }
            int v = b.signal.get();
            if (b.mode == HYD_CLASS_) {
                if (v != 0) { dom_class_add(b.handle, b.name.as_ptr()); }
                else { dom_class_remove(b.handle, b.name.as_ptr()); }
            } else if (b.mode == HYD_VISIBLE_) {
                dom_set_visible(b.handle, v != 0);
            } else if (b.mode == HYD_CHECKED_) {
                dom_set_checked(b.handle, v != 0);
            }
        }
    }
}

static ?&HydrateStrBinding hydrate_str_new_(&Signal<String> signal, int handle, int mode, const &char name) {
    ?&HydrateStrBinding bq = null;
    unsafe { bq = (?&HydrateStrBinding)alloc_raw_(sizeof(struct HydrateStrBinding)); }
    match (bq) {
        case null: { }
        case some(b): {
            b.signal = signal;
            b.handle = handle;
            b.mode = mode;
            b.name = String::from(name);
            b.writing = 0;
        }
    }
    hydrate_str_apply_(bq);
    signal_subscribe(signal, hydrate_str_apply_, bq);
    return bq;
}

static ?&HydrateIntBinding hydrate_int_new_(&Signal<int> signal, int handle, int mode, const &char name) {
    ?&HydrateIntBinding bq = null;
    unsafe { bq = (?&HydrateIntBinding)alloc_raw_(sizeof(struct HydrateIntBinding)); }
    match (bq) {
        case null: { }
        case some(b): {
            b.signal = signal;
            b.handle = handle;
            b.mode = mode;
            b.name = String::from(name);
            b.writing = 0;
        }
    }
    hydrate_int_apply_(bq);
    signal_subscribe(signal, hydrate_int_apply_, bq);
    return bq;
}

void hydrate_bind_text(&Signal<String> signal, int elementHandle) {
    hydrate_str_new_(signal, elementHandle, HYD_TEXT_, "");
}

void hydrate_bind_attr(&Signal<String> signal, int elementHandle, const &char attr) {
    hydrate_str_new_(signal, elementHandle, HYD_ATTR_, attr);
}

void hydrate_bind_class(&Signal<int> signal, int elementHandle, const &char cls) {
    hydrate_int_new_(signal, elementHandle, HYD_CLASS_, cls);
}

void hydrate_bind_visible(&Signal<int> signal, int elementHandle) {
    hydrate_int_new_(signal, elementHandle, HYD_VISIBLE_, "");
}

static void hydrate_on_input_(?&HydrateStrBinding bq, &DomEvent ev) {
    match (bq) {
        case null: { }
        case some(b): {
            struct String v = ev.target_value();
            &String cur = b.signal.get_ref();
            if (!cur.eq(&v)) {
                b.writing = 1;
                cur.free();
                *cur = v;
                b.signal.version = b.signal.version + 1UL;
                b.signal.notify();
                b.writing = 0;
            } else {
                v.free();
            }
        }
    }
}

int hydrate_bind_value(&Signal<String> signal, int elementHandle) {
    ?&HydrateStrBinding bq = hydrate_str_new_(signal, elementHandle, HYD_VALUE_, "");
    return dom_on(elementHandle, "input", hydrate_on_input_, bq);
}

static void hydrate_on_change_(?&HydrateIntBinding bq, &DomEvent ev) {
    match (bq) {
        case null: { }
        case some(b): {
            int v = ev.target_checked();
            if (b.signal.get() != v) {
                b.writing = 1;
                b.signal.set(v);
                b.writing = 0;
            }
        }
    }
}

int hydrate_bind_checked(&Signal<int> signal, int elementHandle) {
    ?&HydrateIntBinding bq = hydrate_int_new_(signal, elementHandle, HYD_CHECKED_, "");
    return dom_on(elementHandle, "change", hydrate_on_change_, bq);
}

int hydrate_bind_text_int(&Signal<int> signal, const &char selector) {
    int handle = dom_get_element(selector);
    if (handle < 0) {
        return -1;
    }
    hydrate_bind_text_int_handle(signal, handle);
    return handle;
}

} // namespace std
