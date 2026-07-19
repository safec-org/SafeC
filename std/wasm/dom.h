#pragma once
// SafeC Standard Library — wasm32 DOM interop.
//
// The JS-facing half of client hydration: a wasm32-unknown-unknown
// module has no DOM access itself, so every DOM operation is an
// 'extern' function that becomes a WASM import (see wasm_rt.h's header
// comment on '-nostdlib'/'--allow-undefined' — an undefined extern
// becomes an import resolved from the host's importObject.env at
// WebAssembly.instantiate() time, under the default "env" module name
// LLVM/lld give unresolved externs). Strings cross the boundary as a
// (ptr, len) pair into the module's own linear memory — the host reads
// them via a Uint8Array view over 'instance.exports.memory.buffer' (see
// std/wasm/HYDRATION.md for the JS-side glue this implies); there is no
// ownership transfer, the host must finish reading before the wasm side
// reuses/frees that memory (true here because every dom_* call below is
// synchronous — the host callback returns before SafeC's caller resumes).
//
// Only a click-event / text-content / attribute surface is provided —
// enough for a real interactive hydrated widget (a counter, a toggle,
// a form field) without pulling in a full DOM binding generator, which
// is out of scope here the same way std/gui/gui_svg.sc's SVG subset and
// std/gui/gui_png.sc's color-type subset are: documented, deliberate
// cuts, not oversights.
namespace std {

// ── Raw JS imports (defined host-side, not in any .sc file) ────────────────
extern int  js_get_element(const char* selPtr, unsigned long selLen);
extern void js_set_text(int handle, const char* textPtr, unsigned long textLen);
extern void js_set_attr(int handle, const char* namePtr, unsigned long nameLen,
                         const char* valPtr, unsigned long valLen);
extern int  js_add_click_listener(int handle, int callbackId);
extern void js_console_log(const char* ptr, unsigned long len);

// ── Ergonomic (NUL-terminated const char*) wrappers ─────────────────────────

// Length of a NUL-terminated string. No libc 'strlen' on a freestanding
// wasm32 build (see wasm_rt.h) — this is the whole reason dom.sc carries
// its own tiny copy rather than declaring 'extern unsigned long strlen(...)'.
unsigned long wasm_strlen(const char* s);

// Looks up the first element matching the CSS-selector-shaped string
// 'selector' (interpretation is entirely host-side — see the JS glue).
// Returns an opaque handle, or -1 if nothing matched.
int  dom_get_element(const char* selector);

void dom_set_text(int handle, const char* text);
void dom_set_attr(int handle, const char* name, const char* value);
void dom_console_log(const char* text);

typedef fn void(void* userdata) DomEventHandler;

// Registers 'handler'/'userdata' as a callback slot and returns its id
// (stable for the module's lifetime). Up to DOM_MAX_CALLBACKS live
// registrations — matches the fixed-table convention used by
// std/rpc/server_fn.h's SERVER_FN_MAX and std/gui/gui_widget.h's
// callback fields (SafeC closures have no captured-environment heap
// allocation, so a fixed dispatch table is the standard workaround
// throughout std, not something new to this file).
#define DOM_MAX_CALLBACKS 32
int  dom_register_callback(DomEventHandler handler, void* userdata);

// Looks up 'selector' and wires a click listener to 'handler'/'userdata'
// via dom_register_callback() + js_add_click_listener(). Returns the
// element handle (as dom_get_element() would), or -1 if not found (no
// listener registered in that case).
int  dom_on_click(const char* selector, DomEventHandler handler, void* userdata);

// Called BY THE HOST (exported at link time via
// '-Wl,--export=std_dom_dispatch_event' — note the 'std_' prefix: a
// function *defined* inside 'namespace std' mangles to 'std_<name>',
// unlike an 'extern' declaration or a top-level definition, which both
// keep their bare name — see wasm_rt.sc's header comment for the same
// distinction applied to malloc/free/etc), never by SafeC code itself:
// looks up 'callbackId' in the registration table and invokes its handler.
void dom_dispatch_event(int callbackId);

// Minimal signed-decimal integer formatter — no snprintf (see
// wasm_strlen's comment; same reason). 'buf' must be at least 24 bytes
// (fits any 64-bit value's sign + digits + NUL). Always NUL-terminates.
void wasm_itoa(long long v, char* buf, unsigned long cap);

} // namespace std
