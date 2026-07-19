// SafeC Standard Library — WASM client hydration implementation (see
// hydrate.h).
#pragma once
#include <std/wasm/hydrate.h>
#include <std/wasm/dom.h>
#include <std/wasm/wasm_rt.h>
#include <std/reactive/signal.h>

namespace std {

struct HydrateTextBinding {
    struct Signal* signal;
    int elementHandle;
};

static void __hydrate_text_int_listener(void* userdata) {
    struct HydrateTextBinding* b;
    unsafe { b = (struct HydrateTextBinding*)userdata; }
    struct Signal* s;
    int handle;
    unsafe { s = b->signal; handle = b->elementHandle; }

    int* valPtr;
    unsafe { valPtr = (int*)signal_get_raw(s); }
    if (valPtr == (int*)0) {
        return;
    }
    int val;
    unsafe { val = *valPtr; }

    char buf[24];
    unsafe { wasm_itoa((long long)val, buf, 24UL); }
    unsafe { dom_set_text(handle, (const char*)buf); }
}

void hydrate_bind_text_int_handle(struct Signal* signal, int elementHandle) {
    struct HydrateTextBinding* b;
    unsafe { b = (struct HydrateTextBinding*)malloc(sizeof(struct HydrateTextBinding)); }
    if (b == (struct HydrateTextBinding*)0) {
        return;
    }
    unsafe {
        b->signal = signal;
        b->elementHandle = elementHandle;
    }
    unsafe { signal->subscribe(__hydrate_text_int_listener, (void*)b); }
}

int hydrate_bind_text_int(struct Signal* signal, const char* selector) {
    int handle = dom_get_element(selector);
    if (handle < 0) {
        return -1;
    }
    hydrate_bind_text_int_handle(signal, handle);
    return handle;
}

} // namespace std
