#pragma once
// SafeC Standard Library — wasm32 DOM and browser-API interop.
//
// A wasm32-unknown-unknown module has no access to the page itself, so
// every browser operation here is an 'extern' js_* function that becomes a
// WebAssembly import (an undefined extern in a module linked with
// --allow-undefined is imported from "env"), implemented by the JS glue
// that ships beside this header: std/wasm/safec_dom.js. Load a module
// through it —
//
//   import { loadSafeC } from "./safec_dom.js";
//   const app = await loadSafeC(fetch("/app.wasm"));   // instantiates, wires every import
//   app.exports.start();
//
// — and every dom_* function below works; see safec_dom.js's header for
// running under Node with a fake document (that is how the tests run it).
//
// ABI. Strings go to JS as (pointer, length) into the module's linear
// memory and are read synchronously. Strings come back through a caller
// buffer: the import copies at most 'cap' bytes and returns the full
// length (-1 for "absent"), and the wrappers here retry with a buffer of
// that size, so the dom_* API hands out ordinary owned Strings. Elements
// are integer handles into a JS-side table (-1: none); DOM_DOCUMENT and
// DOM_WINDOW name the document and window. A handle stays valid until
// dom_release() — release elements you looked up or created in a loop,
// or the table grows for the page's lifetime.
//
// Callbacks (events, timers, animation frames, fetch completions) are
// slots in one table of DOM_MAX_CALLBACKS entries; JS invokes them by id
// through the exported std_dom_dispatch_* functions. A handler receives
// its userdata (the generic ?&T it was registered with) and, for events, a
// DomEvent whose accessors read the event being dispatched — valid only
// during the handler. Every wrapper pairs fn void(?&T, ...) with ?&T, so
// the type-erased storage is always called correctly.
//
// Memory: the strings these wrappers return are heap Strings; free them.
// (std/wasm/wasm_rt.sc's allocator is a bump allocator whose free() is a
// no-op — a long-running page should keep per-event allocation small.)
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

// ── Raw JS imports (implemented by std/wasm/safec_dom.js) ──────────────────
// Original surface (kept for existing glue).
extern int  js_get_element(const char* selPtr, unsigned long selLen);
extern void js_set_text(int handle, const char* textPtr, unsigned long textLen);
extern void js_set_attr(int handle, const char* namePtr, unsigned long nameLen,
                         const char* valPtr, unsigned long valLen);
extern int  js_add_click_listener(int handle, int callbackId);
extern void js_console_log(const char* ptr, unsigned long len);
// Elements and tree.
extern int    js_query(int parent, const char* sel, int selLen);
extern int    js_query_all(int parent, const char* sel, int selLen, int* out, int cap);
extern int    js_create_element(const char* tag, int tagLen);
extern int    js_create_text(const char* text, int textLen);
extern void   js_append_child(int parent, int child);
extern void   js_insert_before(int parent, int child, int ref);
extern void   js_remove(int handle);
extern void   js_release(int handle);
extern int    js_relative(int handle, int which);
extern int    js_children(int handle, int* out, int cap);
// Content, attributes, classes, style, properties.
extern int    js_get_text(int handle, char* buf, int cap);
extern void   js_set_html(int handle, const char* html, int len);
extern int    js_get_html(int handle, char* buf, int cap);
extern int    js_get_attr(int handle, const char* name, int nameLen, char* buf, int cap);
extern void   js_remove_attr(int handle, const char* name, int nameLen);
extern int    js_class_op(int handle, int op, const char* cls, int clsLen);
extern void   js_set_style(int handle, const char* prop, int propLen, const char* val, int valLen);
extern int    js_get_prop(int handle, const char* name, int nameLen, char* buf, int cap);
extern void   js_set_prop(int handle, const char* name, int nameLen, const char* val, int valLen);
extern double js_get_prop_num(int handle, const char* name, int nameLen);
extern void   js_set_prop_num(int handle, const char* name, int nameLen, double val);
extern void   js_call(int handle, const char* method, int methodLen);
// Events.
extern int    js_add_listener(int handle, const char* type, int typeLen, int callbackId, int flags);
extern void   js_remove_listener(int listenerId);
extern double js_event_num(int field);
extern int    js_event_str(int field, char* buf, int cap);
extern void   js_event_control(int op);
// Timers and time.
extern int    js_set_timer(int callbackId, int ms, int repeat);
extern void   js_clear_timer(int timerId);
extern int    js_request_frame(int callbackId);
extern void   js_cancel_frame(int frameId);
extern double js_now();
// fetch().
extern int    js_fetch(int callbackId, const char* method, int methodLen, const char* url, int urlLen,
                       const char* headers, int headersLen, const char* body, int bodyLen);
extern int    js_fetch_str(int requestId, int field, char* buf, int cap);
extern int    js_fetch_header(int requestId, const char* name, int nameLen, char* buf, int cap);
extern void   js_fetch_done(int requestId);
// localStorage / sessionStorage.
extern int    js_storage_get(int kind, const char* key, int keyLen, char* buf, int cap);
extern int    js_storage_set(int kind, const char* key, int keyLen, const char* val, int valLen);
extern void   js_storage_remove(int kind, const char* key, int keyLen);
extern void   js_storage_clear(int kind);
extern int    js_storage_length(int kind);
extern int    js_storage_key(int kind, int index, char* buf, int cap);
// location / history / console.
extern int    js_location(int field, char* buf, int cap);
extern void   js_history(int op, const char* url, int urlLen);
extern void   js_console(int level, const char* ptr, int len);

// ── Constants ──────────────────────────────────────────────────────────────

#define DOM_NONE      (-1)
#define DOM_DOCUMENT  (-2)   // handle of 'document'
#define DOM_WINDOW    (-3)   // handle of 'window' (popstate, resize, keydown, ...)

#define DOM_LOG_DEBUG 0
#define DOM_LOG_INFO  1
#define DOM_LOG_LOG   2
#define DOM_LOG_WARN  3
#define DOM_LOG_ERROR 4

#define DOM_STORAGE_LOCAL   0
#define DOM_STORAGE_SESSION 1

// dom_location() fields.
#define DOM_LOC_HREF     0
#define DOM_LOC_PATHNAME 1
#define DOM_LOC_SEARCH   2
#define DOM_LOC_HASH     3
#define DOM_LOC_HOST     4
#define DOM_LOC_ORIGIN   5

// dom_add_listener flags.
#define DOM_LISTEN_CAPTURE 1
#define DOM_LISTEN_ONCE    2
#define DOM_LISTEN_PASSIVE 4

// DomEvent fields (js_event_num / js_event_str).
#define DOM_EV_TARGET        0    // handle of event.target
#define DOM_EV_CLIENT_X      1
#define DOM_EV_CLIENT_Y      2
#define DOM_EV_BUTTON        3
#define DOM_EV_BUTTONS       4
#define DOM_EV_MODIFIERS     5    // bit 0 alt, 1 ctrl, 2 shift, 3 meta
#define DOM_EV_DELTA_X       6
#define DOM_EV_DELTA_Y       7
#define DOM_EV_REPEAT        8
#define DOM_EV_TIMESTAMP     9
#define DOM_EV_OFFSET_X      10
#define DOM_EV_OFFSET_Y      11
#define DOM_EV_TARGET_CHECKED 12
#define DOM_EV_CURRENT_TARGET 13
#define DOM_EV_STR_TYPE      0
#define DOM_EV_STR_KEY       1
#define DOM_EV_STR_CODE      2
#define DOM_EV_STR_VALUE     3    // event.target.value
#define DOM_EV_STR_DATA      4    // InputEvent.data / MessageEvent.data
#define DOM_EV_STR_STATE     5    // popstate: JSON.stringify(event.state)

#define DOM_MOD_ALT   1
#define DOM_MOD_CTRL  2
#define DOM_MOD_SHIFT 4
#define DOM_MOD_META  8

// ── Strings ────────────────────────────────────────────────────────────────

// Length of a NUL-terminated string. No libc 'strlen' on a freestanding
// wasm32 build — this is the whole reason dom.sc carries its own copy.
unsigned long wasm_strlen(const &char s);

// Minimal signed-decimal integer formatter — no snprintf on wasm32. 'buf'
// must be at least 24 bytes (fits any 64-bit value's sign + digits + NUL).
// Always NUL-terminates.
void wasm_itoa(long long v, &char buf, unsigned long cap);

// ── Elements ───────────────────────────────────────────────────────────────

// First element matching the CSS selector (document.querySelector);
// -1 if none. dom_query is the same; dom_query_in searches under 'parent'.
int  dom_get_element(const &char selector);
int  dom_query(const &char selector);
int  dom_query_in(int parent, const &char selector);
// document.getElementById.
int  dom_get_by_id(const &char id);
// Every match (querySelectorAll), appended to 'out' as handles; returns
// the count.
unsigned long dom_query_all(const &char selector, &Vec<int> out);
unsigned long dom_query_all_in(int parent, const &char selector, &Vec<int> out);
// document.body / document.head.
int  dom_body();
int  dom_head();

// document.createElement / createTextNode.
int  dom_create_element(const &char tag);
int  dom_create_text(const &char text);
// parent.appendChild(child); parent.insertBefore(child, ref) (ref -1:
// append); node.remove().
void dom_append(int parent, int child);
void dom_insert_before(int parent, int child, int ref);
void dom_remove(int handle);
// Removes every child (textContent = "").
void dom_clear(int handle);
// Drops the handle (the element itself is untouched).
void dom_release(int handle);
// parentElement / firstElementChild / nextElementSibling /
// previousElementSibling / lastElementChild, or -1.
int  dom_parent(int handle);
int  dom_first_child(int handle);
int  dom_next_sibling(int handle);
int  dom_prev_sibling(int handle);
int  dom_last_child(int handle);
// The element children (handles appended to 'out'); returns the count.
unsigned long dom_children(int handle, &Vec<int> out);

// ── Content ────────────────────────────────────────────────────────────────

void dom_set_text(int handle, const &char text);       // textContent
struct String dom_get_text(int handle);
void dom_set_html(int handle, const &char html);       // innerHTML (trusted markup only)
struct String dom_get_html(int handle);

void dom_set_attr(int handle, const &char name, const &char value);
// The attribute's value; empty if absent (dom_has_attr tells the two apart).
struct String dom_get_attr(int handle, const &char name);
int  dom_has_attr(int handle, const &char name);
void dom_remove_attr(int handle, const &char name);

// classList.add / remove / toggle (returns the new state) / contains.
void dom_class_add(int handle, const &char cls);
void dom_class_remove(int handle, const &char cls);
int  dom_class_toggle(int handle, const &char cls);
int  dom_class_contains(int handle, const &char cls);

// style.setProperty(prop, value) ("" removes it); CSS property names
// ("background-color"), not camelCase.
void dom_set_style(int handle, const &char prop, const &char value);
// Shows/hides via style.display ("" / "none").
void dom_set_visible(int handle, int visible);

// Form controls: .value, .checked, .disabled.
struct String dom_get_value(int handle);
void dom_set_value(int handle, const &char value);
int  dom_get_checked(int handle);
void dom_set_checked(int handle, int checked);
void dom_set_disabled(int handle, int disabled);

// Any property: String(el[name]) (empty for undefined/null), el[name] =
// value; the numeric forms convert with Number() / assign a number (0/1
// works for boolean properties). DOM_DOCUMENT and DOM_WINDOW work too:
// dom_set_prop(DOM_DOCUMENT, "title", "...").
struct String dom_get_prop(int handle, const &char name);
// The same into a caller buffer: copies up to cap - 1 bytes, NUL-
// terminates, returns the full length (-1 if undefined/null).
int  dom_get_prop_buf(int handle, const &char name, &char buf, int cap);
void dom_set_prop(int handle, const &char name, const &char value);
double dom_get_prop_num(int handle, const &char name);
void dom_set_prop_num(int handle, const &char name, double value);
// el[method]() with no arguments: "focus", "blur", "click", "select",
// "scrollIntoView", "reset", "submit", ...
void dom_call(int handle, const &char method);
void dom_focus(int handle);

// ── Console ────────────────────────────────────────────────────────────────

void dom_console_log(const &char text);
// console.debug/info/log/warn/error by DOM_LOG_* level.
void dom_console(int level, const &char text);

// ── Callback table ─────────────────────────────────────────────────────────

// Stored type-erased (the table holds handlers with unrelated context
// types); the generic wrappers below are the only writers and always pair
// a handler with userdata of the matching type. A callback id is
// slot + DOM_MAX_CALLBACKS * generation, so an id whose slot was freed
// and reused (a timer that already fired, say) is recognized as stale.
typedef fn void(?&Erased userdata) DomEventHandler;   // erased, see std/erased.h

#define DOM_MAX_CALLBACKS 256

#define DOM_CB_PLAIN_ 0    // fn void(?&T)
#define DOM_CB_EVENT_ 1    // fn void(?&T, &DomEvent)
#define DOM_CB_FRAME_ 2    // fn void(?&T, double)
#define DOM_CB_FETCH_ 3    // fn void(?&T, &DomFetchResponse)

// Frees a callback slot (dom_off / dom_clear_timer / ... do this).
void dom_unregister_callback(int callbackId);

// Registers handler(userdata) as a callback slot and returns its id (-1 if
// the table is full). Stays registered until dom_unregister_callback().
int  dom_register_callback_raw_(fn void(?&Erased) handler, ?&Erased userdata);
generic<T>
int  dom_register_callback(fn void(?&T) handler, ?&T userdata) {
    unsafe { return dom_register_callback_raw_((fn void(?&Erased))handler, (?&Erased)userdata); }
}

// Looks up 'selector' and calls handler(userdata) on every click. Returns
// the element handle, or -1 if not found (no listener registered then).
int  dom_on_click_raw_(const &char selector, fn void(?&Erased) handler, ?&Erased userdata);
generic<T>
int  dom_on_click(const &char selector, fn void(?&T) handler, ?&T userdata) {
    unsafe { return dom_on_click_raw_(selector, (fn void(?&Erased))handler, (?&Erased)userdata); }
}

// ── Events ─────────────────────────────────────────────────────────────────

// The event being dispatched. Accessors call into JS for the live event:
// only valid inside the handler that received it.
struct DomEvent {
    int listener;      // the listener id dom_on returned (its callback id)

    struct String type() const;           // "click", "keydown", ...
    struct String key() const;            // KeyboardEvent.key ("a", "Enter", "ArrowUp")
    struct String code() const;           // KeyboardEvent.code ("KeyA")
    // event.target.value (input/change on form controls), its .checked.
    struct String target_value() const;
    int           target_checked() const;
    struct String data() const;           // InputEvent/MessageEvent data
    struct String state_json() const;     // popstate: JSON of history.state ("null" if none)
    int           target() const;         // handle of event.target (release it when done)
    int           current_target() const;
    double        client_x() const;
    double        client_y() const;
    double        offset_x() const;
    double        offset_y() const;
    int           button() const;
    int           buttons() const;
    double        delta_x() const;        // WheelEvent
    double        delta_y() const;
    int           modifiers() const;      // DOM_MOD_* bits
    int           alt_key() const;
    int           ctrl_key() const;
    int           shift_key() const;
    int           meta_key() const;
    int           repeat() const;
    double        timestamp() const;
    void          prevent_default() const;
    void          stop_propagation() const;
};

// el.addEventListener(type, ...) for any event type; the handler gets the
// userdata and the event. 'flags' is DOM_LISTEN_* bits (0 for none).
// Returns a listener id for dom_off(), or -1 (bad handle / table full).
int  dom_add_listener_raw_(int handle, const &char type, fn void(?&Erased, &DomEvent) handler, ?&Erased userdata, int flags);
generic<T>
int  dom_add_listener(int handle, const &char type, fn void(?&T, &DomEvent) handler, ?&T userdata, int flags) {
    unsafe { return dom_add_listener_raw_(handle, type, (fn void(?&Erased, &DomEvent))handler, (?&Erased)userdata, flags); }
}
generic<T>
int  dom_on(int handle, const &char type, fn void(?&T, &DomEvent) handler, ?&T userdata) {
    unsafe { return dom_add_listener_raw_(handle, type, (fn void(?&Erased, &DomEvent))handler, (?&Erased)userdata, 0); }
}
// dom_on on the first element matching 'selector'; the listener id, or -1.
int  dom_on_selector_raw_(const &char selector, const &char type, fn void(?&Erased, &DomEvent) handler, ?&Erased userdata);
generic<T>
int  dom_on_selector(const &char selector, const &char type, fn void(?&T, &DomEvent) handler, ?&T userdata) {
    unsafe { return dom_on_selector_raw_(selector, type, (fn void(?&Erased, &DomEvent))handler, (?&Erased)userdata); }
}
// removeEventListener + frees the callback slot.
void dom_off(int listenerId);

// ── Timers ─────────────────────────────────────────────────────────────────

// setTimeout / setInterval; the timer id for dom_clear_timer, or -1. A
// one-shot timer's slot is freed after it fires.
int  dom_set_timer_raw_(fn void(?&Erased) handler, ?&Erased userdata, int ms, int repeat);
generic<T>
int  dom_set_timeout(fn void(?&T) handler, ?&T userdata, int ms) {
    unsafe { return dom_set_timer_raw_((fn void(?&Erased))handler, (?&Erased)userdata, ms, 0); }
}
generic<T>
int  dom_set_interval(fn void(?&T) handler, ?&T userdata, int ms) {
    unsafe { return dom_set_timer_raw_((fn void(?&Erased))handler, (?&Erased)userdata, ms, 1); }
}
void dom_clear_timer(int timerId);

// requestAnimationFrame: handler(userdata, timestamp_ms) once, on the next
// frame (re-request from inside the handler for an animation loop). The
// frame id, or -1.
int  dom_request_animation_frame_raw_(fn void(?&Erased, double) handler, ?&Erased userdata);
generic<T>
int  dom_request_animation_frame(fn void(?&T, double) handler, ?&T userdata) {
    unsafe { return dom_request_animation_frame_raw_((fn void(?&Erased, double))handler, (?&Erased)userdata); }
}
void dom_cancel_animation_frame(int frameId);

// performance.now(), in milliseconds.
double dom_now();

// ── fetch() ────────────────────────────────────────────────────────────────

// A completed fetch, as its handler sees it. 'status' is 0 (and ok 0) on
// a network error; text() etc. are only valid inside the handler.
struct DomFetchResponse {
    int request;
    int status;
    int ok;            // status 200-299

    struct String text() const;           // the body as text
    struct String status_text() const;
    struct String url() const;            // final URL (after redirects)
    struct String header(const &char name) const;   // empty if absent
};

// Starts fetch(url, {method, headers, body}); 'headers' is null or
// "Name: value" lines separated by "\r\n" or "\n"; 'body' null for none.
// handler(userdata, &response) runs once it completes (or fails). Returns
// the request id, or -1.
int  dom_fetch_raw_(const &char method, const &char url, const ?&char headers, const ?&char body,
                    fn void(?&Erased, &DomFetchResponse) handler, ?&Erased userdata);
generic<T>
int  dom_fetch(const &char method, const &char url, const ?&char headers, const ?&char body,
               fn void(?&T, &DomFetchResponse) handler, ?&T userdata) {
    unsafe { return dom_fetch_raw_(method, url, headers, body, (fn void(?&Erased, &DomFetchResponse))handler, (?&Erased)userdata); }
}
generic<T>
int  dom_fetch_get(const &char url, fn void(?&T, &DomFetchResponse) handler, ?&T userdata) {
    unsafe { return dom_fetch_raw_("GET", url, null, null, (fn void(?&Erased, &DomFetchResponse))handler, (?&Erased)userdata); }
}

// ── Storage ────────────────────────────────────────────────────────────────

// localStorage (DOM_STORAGE_LOCAL) / sessionStorage (DOM_STORAGE_SESSION).
// get: 1 and the value appended to 'out', or 0 if the key is absent.
int  dom_storage_get(int kind, const &char key, &String out);
// 1 on success, 0 if the browser refused (quota, storage disabled).
int  dom_storage_set(int kind, const &char key, const &char value);
void dom_storage_remove(int kind, const &char key);
void dom_storage_clear(int kind);
int  dom_storage_length(int kind);
// The i-th key (empty past the end).
struct String dom_storage_key(int kind, int index);

// ── Location and history (client-side routing) ────────────────────────────

// A location field (DOM_LOC_*): href, pathname, search ("?a=1"), hash
// ("#x"), host, origin.
struct String dom_location(int field);
struct String dom_location_path();
// history.pushState(null, "", url) / replaceState / back() / forward().
// Listen for DOM_WINDOW "popstate" to react to back/forward.
void dom_history_push(const &char url);
void dom_history_replace(const &char url);
void dom_history_back();
void dom_history_forward();
// location.assign(url): a real navigation.
void dom_navigate(const &char url);

// ── Host-called entry points ───────────────────────────────────────────────
// Exported for the JS glue, never called by SafeC code. Names are fixed
// (std_dom_dispatch_event is the original export existing glue uses).
__attribute__((export_name("std_dom_dispatch_event")))
void dom_dispatch_event(int callbackId);
__attribute__((export_name("std_dom_dispatch_frame")))
void dom_dispatch_frame(int callbackId, double timestamp);
__attribute__((export_name("std_dom_dispatch_fetch")))
void dom_dispatch_fetch(int callbackId, int requestId, int status);

} // namespace std
