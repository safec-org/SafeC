#pragma once
// SafeC Standard Library — WASM client hydration.
//
// The piece that actually ties std/reactive/signal.h's Signal<T> to
// std/wasm/dom.h's DOM bindings: fine-grained, SolidJS-style hydration —
// a Signal knows exactly which DOM node(s) depend on it (via subscribe())
// and pushes updates directly, no virtual DOM / diff / re-render-the-tree
// step. A hydrated page is server-rendered scx markup (see
// std/scx/scx.h) whose interactive elements the client-side wasm module
// locates by selector and wires up on startup — see
// examples/fullstack_demo for the full round trip (server render +
// server function + hydrated client interactivity).
#include <std/reactive/signal.h>
#include <std/wasm/dom.h>
#include <std/collections/string.h>

namespace std {

// Binds 'signal' (must hold an 'int') so every future change formats its
// value as decimal text and writes it into 'elementHandle's text content.
// Does not set the element's *initial* text — call dom_set_text()
// yourself first (or set() once) if it needs to show a value
// before the first change.
void hydrate_bind_text_int_handle(&Signal<int> signal, int elementHandle);

// Convenience: dom_get_element(selector) + hydrate_bind_text_int_handle().
// Returns the resolved element handle (-1 if 'selector' matched nothing,
// in which case no binding was registered).
int hydrate_bind_text_int(&Signal<int> signal, const &char selector);

// The bindings below write the signal's current value right away, then
// again on every change (unlike the two _int ones above, which only react
// to changes). Each allocates a small binding record that lives as long as
// the page (signals have no unsubscribe-one).

// Text content from a Signal<String>.
void hydrate_bind_text(&Signal<String> signal, int elementHandle);
// An attribute's value from a Signal<String>.
void hydrate_bind_attr(&Signal<String> signal, int elementHandle, const &char attr);
// Adds 'cls' to the element's classList while the signal is non-zero,
// removes it otherwise.
void hydrate_bind_class(&Signal<int> signal, int elementHandle, const &char cls);
// Shows the element while the signal is non-zero (style.display), hides it
// otherwise.
void hydrate_bind_visible(&Signal<int> signal, int elementHandle);
// Two-way binding for an <input>/<textarea>/<select>: the control's value
// follows the signal, and every "input" event stores the control's value
// into the signal (which notifies the signal's other subscribers).
// Returns the listener id (dom_off() stops the DOM -> signal half).
int  hydrate_bind_value(&Signal<String> signal, int elementHandle);
// Two-way binding of a checkbox's .checked to a Signal<int> ("change").
int  hydrate_bind_checked(&Signal<int> signal, int elementHandle);

// Text content from any Signal<T>, rendered by 'format' (which appends the
// text for a value to 'out').
generic<T>
struct HydrateFmtBinding {
    &Signal<T> signal;
    int        handle;
    fn void(&String, const &T) format;
};

generic<T>
void hydrate_fmt_apply_(?&HydrateFmtBinding<T> bq) {
    match (bq) {
        case null: { }
        case some(b): {
            struct String text = String::new();
            T v = b.signal.get();
            b.format(&text, &v);
            dom_set_text(b.handle, text.as_ptr());
            text.free();
        }
    }
}

generic<T>
void hydrate_bind_text_with(&Signal<T> signal, int elementHandle, fn void(&String, const &T) format) {
    ?&HydrateFmtBinding<T> bq = null;
    unsafe { bq = (?&HydrateFmtBinding<T>)alloc_raw_(sizeof(struct HydrateFmtBinding<T>)); }
    match (bq) {
        case null: { return; }
        case some(b): {
            b.signal = signal;
            b.handle = elementHandle;
            b.format = format;
        }
    }
    hydrate_fmt_apply_(bq);
    signal_subscribe(signal, hydrate_fmt_apply_, bq);
}

} // namespace std
