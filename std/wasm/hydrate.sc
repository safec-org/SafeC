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
                dom_set_text(b.elementHandle, (const char*)buf);
            }
        }
    }
}

void hydrate_bind_text_int_handle(&Signal<int> signal, int elementHandle) {
    ?&HydrateTextBinding bq = null;
    unsafe { bq = (?&HydrateTextBinding)alloc(sizeof(struct HydrateTextBinding)); }
    match (bq) {
        case null: return;
        case some(b): {
            b.signal = signal;
            b.elementHandle = elementHandle;
        }
    }
    signal_subscribe(signal, __hydrate_text_int_listener, bq);
}

int hydrate_bind_text_int(&Signal<int> signal, const char* selector) {
    int handle = dom_get_element(selector);
    if (handle < 0) {
        return -1;
    }
    hydrate_bind_text_int_handle(signal, handle);
    return handle;
}

} // namespace std
