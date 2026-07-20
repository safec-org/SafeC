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

namespace std {

// Binds 'signal' (must hold an 'int') so every future change formats its
// value as decimal text and writes it into 'elementHandle's text content.
// Does not set the element's *initial* text — call dom_set_text()
// yourself first (or signal_set_t() once) if it needs to show a value
// before the first change.
void hydrate_bind_text_int_handle(&stack Signal signal, int elementHandle);

// Convenience: dom_get_element(selector) + hydrate_bind_text_int_handle().
// Returns the resolved element handle (-1 if 'selector' matched nothing,
// in which case no binding was registered).
int hydrate_bind_text_int(&stack Signal signal, const char* selector);

} // namespace std
