// SafeC Standard Library — wasm32 DOM interop implementation (see dom.h).
#pragma once
#include <std/wasm/dom.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/mem.h>

namespace std {

#define DOM_KIND_TIMER_ 16    // plain callback used by a timer (slot.extra = JS timer id)
#define DOM_KIND_ONESHOT_ 32  // freed after its first dispatch

struct DomCallbackSlot {
    void* handler;
    void* userdata;
    int   kind;       // DOM_CB_* | DOM_KIND_* bits
    int   used;
    int   gen;        // bumped every time the slot is freed
    int   extra;      // JS-side id (listener / timer / frame / request)
};

static struct DomCallbackSlot gDomCallbacks[DOM_MAX_CALLBACKS];

// ── Strings ────────────────────────────────────────────────────────────────

unsigned long wasm_strlen(const &char s) {
    unsigned long n = 0UL;
    unsafe {
        while (s[n] != (char)0) {
            n = n + 1UL;
        }
    }
    return n;
}

static int dom_len_(const &char s) { return (int)wasm_strlen(s); }

void wasm_itoa(long long v, &char buf, unsigned long cap) {
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

// One string-returning import, described so dom_str_ can call it twice
// (a stack buffer first, an exact-size heap one if that was too small).
#define DOM_STR_TEXT_     0
#define DOM_STR_HTML_     1
#define DOM_STR_ATTR_     2
#define DOM_STR_PROP_     3
#define DOM_STR_EVENT_    4
#define DOM_STR_FETCH_    5
#define DOM_STR_FHEADER_  6
#define DOM_STR_STORAGE_  7
#define DOM_STR_SKEY_     8
#define DOM_STR_LOCATION_ 9

struct DomStrReq_ {
    int op;
    int a;
    int b;
    const char* s;
    int slen;
};

static int dom_str_call_(&DomStrReq_ r, char* buf, int cap) {
    unsafe {
        if (r.op == DOM_STR_TEXT_) { return js_get_text(r.a, buf, cap); }
        if (r.op == DOM_STR_HTML_) { return js_get_html(r.a, buf, cap); }
        if (r.op == DOM_STR_ATTR_) { return js_get_attr(r.a, r.s, r.slen, buf, cap); }
        if (r.op == DOM_STR_PROP_) { return js_get_prop(r.a, r.s, r.slen, buf, cap); }
        if (r.op == DOM_STR_EVENT_) { return js_event_str(r.a, buf, cap); }
        if (r.op == DOM_STR_FETCH_) { return js_fetch_str(r.a, r.b, buf, cap); }
        if (r.op == DOM_STR_FHEADER_) { return js_fetch_header(r.a, r.s, r.slen, buf, cap); }
        if (r.op == DOM_STR_STORAGE_) { return js_storage_get(r.a, r.s, r.slen, buf, cap); }
        if (r.op == DOM_STR_SKEY_) { return js_storage_key(r.a, r.b, buf, cap); }
        if (r.op == DOM_STR_LOCATION_) { return js_location(r.a, buf, cap); }
    }
    return -1;
}

// Appends the result to 'out'; returns 0 when JS reported "absent" (-1).
static int dom_str_into_(&DomStrReq_ r, &String out) {
    char small[256];
    int n;
    unsafe { n = dom_str_call_(r, &small[0], 256); }
    if (n < 0) { return 0; }
    if (n <= 256) {
        unsafe { out.push_n((const char*)&small[0], (unsigned long)n); }
        return 1;
    }
    unsafe {
        char* big = (char*)alloc_raw_((unsigned long)n);
        if (big == (char*)0) { return 1; }
        int m = dom_str_call_(r, big, n);
        if (m > n) { m = n; }
        if (m > 0) { out.push_n((const char*)big, (unsigned long)m); }
        dealloc_raw_((void*)big);
    }
    return 1;
}

static struct String dom_str_(int op, int a, int b, const char* s, int slen) {
    struct DomStrReq_ r;
    r.op = op;
    r.a = a;
    r.b = b;
    r.s = s;
    r.slen = slen;
    struct String out = String::new();
    dom_str_into_(&r, &out);
    return out;
}

// ── Elements ───────────────────────────────────────────────────────────────

int dom_get_element(const &char selector) {
    unsigned long len = wasm_strlen(selector);
    int handle;
    unsafe { handle = js_get_element(selector, len); }
    return handle;
}

int dom_query(const &char selector) {
    unsafe { return js_query(DOM_DOCUMENT, selector, dom_len_(selector)); }
}

int dom_query_in(int parent, const &char selector) {
    unsafe { return js_query(parent, selector, dom_len_(selector)); }
}

int dom_get_by_id(const &char id) {
    struct String sel = String::from("#");
    // CSS-escape is not needed for the usual [A-Za-z0-9_-] ids; anything
    // else goes through an attribute selector.
    int plain = 1;
    unsafe {
        unsigned long i = 0UL;
        while (id[i] != (char)0) {
            char c = id[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) {
                plain = 0;
            }
            i = i + 1UL;
        }
        if (id[0] >= '0' && id[0] <= '9') { plain = 0; }
    }
    if (plain) {
        sel.push(id);
    } else {
        sel.clear();
        sel.push("[id=\"");
        unsafe {
            unsigned long i = 0UL;
            while (id[i] != (char)0) {
                if (id[i] == '"' || id[i] == '\\') { sel.push_char('\\'); }
                sel.push_char(id[i]);
                i = i + 1UL;
            }
        }
        sel.push("\"]");
    }
    int h = dom_query(sel.as_ptr());
    sel.free();
    return h;
}

static unsigned long dom_collect_(int parent, const char* sel, int selLen, int children, &Vec<int> out) {
    int small[64];
    int n;
    unsafe {
        if (children) { n = js_children(parent, &small[0], 64); }
        else { n = js_query_all(parent, sel, selLen, &small[0], 64); }
    }
    if (n <= 0) { return 0UL; }
    if (n <= 64) {
        int i = 0;
        while (i < n) { unsafe { out.push(small[i]); } i = i + 1; }
        return (unsigned long)n;
    }
    // Too many for the stack buffer: the first call already created
    // handles for all of them, so release those and ask again.
    int i = 0;
    while (i < 64) { unsafe { js_release(small[i]); } i = i + 1; }
    unsafe {
        int* big = (int*)alloc_raw_((unsigned long)n * sizeof(int));
        if (big == (int*)0) { return 0UL; }
        int m;
        if (children) { m = js_children(parent, big, n); }
        else { m = js_query_all(parent, sel, selLen, big, n); }
        if (m > n) { m = n; }
        int k = 0;
        while (k < m) { out.push(big[k]); k = k + 1; }
        dealloc_raw_((void*)big);
        return (unsigned long)m;
    }
}

unsigned long dom_query_all(const &char selector, &Vec<int> out) {
    unsafe { return dom_collect_(DOM_DOCUMENT, selector, dom_len_(selector), 0, out); }
}

unsigned long dom_query_all_in(int parent, const &char selector, &Vec<int> out) {
    unsafe { return dom_collect_(parent, selector, dom_len_(selector), 0, out); }
}

unsigned long dom_children(int handle, &Vec<int> out) {
    unsafe { return dom_collect_(handle, (const char*)0, 0, 1, out); }
}

int dom_body() { unsafe { return js_relative(DOM_DOCUMENT, 10); } }
int dom_head() { unsafe { return js_relative(DOM_DOCUMENT, 11); } }

int dom_create_element(const &char tag) {
    unsafe { return js_create_element(tag, dom_len_(tag)); }
}

int dom_create_text(const &char text) {
    unsafe { return js_create_text(text, dom_len_(text)); }
}

void dom_append(int parent, int child) { unsafe { js_append_child(parent, child); } }
void dom_insert_before(int parent, int child, int ref) { unsafe { js_insert_before(parent, child, ref); } }
void dom_remove(int handle) { unsafe { js_remove(handle); } }
void dom_clear(int handle) { unsafe { js_set_text(handle, "", 0UL); } }
void dom_release(int handle) { unsafe { js_release(handle); } }
int  dom_parent(int handle) { unsafe { return js_relative(handle, 0); } }
int  dom_first_child(int handle) { unsafe { return js_relative(handle, 1); } }
int  dom_next_sibling(int handle) { unsafe { return js_relative(handle, 2); } }
int  dom_prev_sibling(int handle) { unsafe { return js_relative(handle, 3); } }
int  dom_last_child(int handle) { unsafe { return js_relative(handle, 4); } }

// ── Content ────────────────────────────────────────────────────────────────

void dom_set_text(int handle, const &char text) {
    unsigned long len = wasm_strlen(text);
    unsafe { js_set_text(handle, text, len); }
}

struct String dom_get_text(int handle) {
    return dom_str_(DOM_STR_TEXT_, handle, 0, (const char*)0, 0);
}

void dom_set_html(int handle, const &char html) {
    unsafe { js_set_html(handle, html, dom_len_(html)); }
}

struct String dom_get_html(int handle) {
    return dom_str_(DOM_STR_HTML_, handle, 0, (const char*)0, 0);
}

void dom_set_attr(int handle, const &char name, const &char value) {
    unsigned long nameLen = wasm_strlen(name);
    unsigned long valLen = wasm_strlen(value);
    unsafe { js_set_attr(handle, name, nameLen, value, valLen); }
}

struct String dom_get_attr(int handle, const &char name) {
    unsafe { return dom_str_(DOM_STR_ATTR_, handle, 0, name, dom_len_(name)); }
}

int dom_has_attr(int handle, const &char name) {
    char one[1];
    int n;
    unsafe { n = js_get_attr(handle, name, dom_len_(name), &one[0], 0); }
    return n >= 0;
}

void dom_remove_attr(int handle, const &char name) {
    unsafe { js_remove_attr(handle, name, dom_len_(name)); }
}

void dom_class_add(int handle, const &char cls) { unsafe { js_class_op(handle, 0, cls, dom_len_(cls)); } }
void dom_class_remove(int handle, const &char cls) { unsafe { js_class_op(handle, 1, cls, dom_len_(cls)); } }
int  dom_class_toggle(int handle, const &char cls) { unsafe { return js_class_op(handle, 2, cls, dom_len_(cls)); } }
int  dom_class_contains(int handle, const &char cls) { unsafe { return js_class_op(handle, 3, cls, dom_len_(cls)); } }

void dom_set_style(int handle, const &char prop, const &char value) {
    unsafe { js_set_style(handle, prop, dom_len_(prop), value, dom_len_(value)); }
}

void dom_set_visible(int handle, int visible) {
    if (visible) { dom_set_style(handle, "display", ""); } else { dom_set_style(handle, "display", "none"); }
}

struct String dom_get_prop(int handle, const &char name) {
    unsafe { return dom_str_(DOM_STR_PROP_, handle, 0, name, dom_len_(name)); }
}

int dom_get_prop_buf(int handle, const &char name, &char buf, int cap) {
    if (cap <= 0) { return 0; }
    int n;
    unsafe {
        n = js_get_prop(handle, name, dom_len_(name), buf, cap - 1);
        int w = n < 0 ? 0 : (n < cap - 1 ? n : cap - 1);
        buf[w] = (char)0;
    }
    return n;
}

void dom_set_prop(int handle, const &char name, const &char value) {
    unsafe { js_set_prop(handle, name, dom_len_(name), value, dom_len_(value)); }
}

double dom_get_prop_num(int handle, const &char name) {
    unsafe { return js_get_prop_num(handle, name, dom_len_(name)); }
}

void dom_set_prop_num(int handle, const &char name, double value) {
    unsafe { js_set_prop_num(handle, name, dom_len_(name), value); }
}

void dom_call(int handle, const &char method) {
    unsafe { js_call(handle, method, dom_len_(method)); }
}

void dom_focus(int handle) { dom_call(handle, "focus"); }

struct String dom_get_value(int handle) { return dom_get_prop(handle, "value"); }
void dom_set_value(int handle, const &char value) { dom_set_prop(handle, "value", value); }
int  dom_get_checked(int handle) { return dom_get_prop_num(handle, "checked") != 0.0; }
void dom_set_checked(int handle, int checked) { dom_set_prop_num(handle, "checked", checked ? 1.0 : 0.0); }
void dom_set_disabled(int handle, int disabled) { dom_set_prop_num(handle, "disabled", disabled ? 1.0 : 0.0); }

// ── Console ────────────────────────────────────────────────────────────────

void dom_console_log(const &char text) {
    unsigned long len = wasm_strlen(text);
    unsafe { js_console_log(text, len); }
}

void dom_console(int level, const &char text) {
    unsafe { js_console(level, text, dom_len_(text)); }
}

// ── Callback table ─────────────────────────────────────────────────────────

int dom_register_callback_kind_raw_(int kind, void* handler, void* userdata) {
    int i = 0;
    while (i < DOM_MAX_CALLBACKS) {
        int used;
        unsafe { used = gDomCallbacks[i].used; }
        if (!used) {
            int gen;
            unsafe {
                gDomCallbacks[i].handler = handler;
                gDomCallbacks[i].userdata = userdata;
                gDomCallbacks[i].kind = kind;
                gDomCallbacks[i].used = 1;
                gDomCallbacks[i].extra = -1;
                if (gDomCallbacks[i].gen <= 0) { gDomCallbacks[i].gen = 1; }
                gen = gDomCallbacks[i].gen;
            }
            return i + DOM_MAX_CALLBACKS * gen;
        }
        i = i + 1;
    }
    return -1;
}

// The slot of a live callback id, or -1.
static int dom_slot_(int id) {
    if (id < 0) { return -1; }
    int slot = id % DOM_MAX_CALLBACKS;
    int gen = id / DOM_MAX_CALLBACKS;
    int ok;
    unsafe { ok = gDomCallbacks[slot].used && gDomCallbacks[slot].gen == gen; }
    return ok ? slot : -1;
}

void dom_unregister_callback(int callbackId) {
    int slot = dom_slot_(callbackId);
    if (slot < 0) { return; }
    unsafe {
        gDomCallbacks[slot].used = 0;
        gDomCallbacks[slot].gen = gDomCallbacks[slot].gen + 1;
        if (gDomCallbacks[slot].gen > 1000000) { gDomCallbacks[slot].gen = 1; }
    }
}

int dom_register_callback_raw_(void* handler, void* userdata) {
    return dom_register_callback_kind_raw_(DOM_CB_PLAIN_, handler, userdata);
}

int dom_on_click_raw_(const &char selector, void* handler, void* userdata) {
    int handle = dom_get_element(selector);
    if (handle < 0) {
        return -1;
    }
    int id = dom_register_callback_raw_(handler, userdata);
    if (id < 0) { return -1; }
    unsafe { js_add_click_listener(handle, id); }
    return handle;
}

// ── Events ─────────────────────────────────────────────────────────────────

struct String DomEvent::type() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_TYPE, 0, (const char*)0, 0); }
struct String DomEvent::key() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_KEY, 0, (const char*)0, 0); }
struct String DomEvent::code() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_CODE, 0, (const char*)0, 0); }
struct String DomEvent::target_value() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_VALUE, 0, (const char*)0, 0); }
struct String DomEvent::data() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_DATA, 0, (const char*)0, 0); }
struct String DomEvent::state_json() const { return dom_str_(DOM_STR_EVENT_, DOM_EV_STR_STATE, 0, (const char*)0, 0); }
static double dom_ev_num_(int f) { unsafe { return js_event_num(f); } }
int    DomEvent::target_checked() const { return dom_ev_num_(DOM_EV_TARGET_CHECKED) != 0.0; }
int    DomEvent::target() const { return (int)dom_ev_num_(DOM_EV_TARGET); }
int    DomEvent::current_target() const { return (int)dom_ev_num_(DOM_EV_CURRENT_TARGET); }
double DomEvent::client_x() const { return dom_ev_num_(DOM_EV_CLIENT_X); }
double DomEvent::client_y() const { return dom_ev_num_(DOM_EV_CLIENT_Y); }
double DomEvent::offset_x() const { return dom_ev_num_(DOM_EV_OFFSET_X); }
double DomEvent::offset_y() const { return dom_ev_num_(DOM_EV_OFFSET_Y); }
int    DomEvent::button() const { return (int)dom_ev_num_(DOM_EV_BUTTON); }
int    DomEvent::buttons() const { return (int)dom_ev_num_(DOM_EV_BUTTONS); }
double DomEvent::delta_x() const { return dom_ev_num_(DOM_EV_DELTA_X); }
double DomEvent::delta_y() const { return dom_ev_num_(DOM_EV_DELTA_Y); }
int    DomEvent::modifiers() const { return (int)dom_ev_num_(DOM_EV_MODIFIERS); }
int    DomEvent::alt_key() const { return (self.modifiers() & DOM_MOD_ALT) != 0; }
int    DomEvent::ctrl_key() const { return (self.modifiers() & DOM_MOD_CTRL) != 0; }
int    DomEvent::shift_key() const { return (self.modifiers() & DOM_MOD_SHIFT) != 0; }
int    DomEvent::meta_key() const { return (self.modifiers() & DOM_MOD_META) != 0; }
int    DomEvent::repeat() const { return dom_ev_num_(DOM_EV_REPEAT) != 0.0; }
double DomEvent::timestamp() const { return dom_ev_num_(DOM_EV_TIMESTAMP); }
void   DomEvent::prevent_default() const { unsafe { js_event_control(0); } }
void   DomEvent::stop_propagation() const { unsafe { js_event_control(1); } }

int dom_add_listener_raw_(int handle, const &char type, void* handler, void* userdata, int flags) {
    int id = dom_register_callback_kind_raw_(DOM_CB_EVENT_ | ((flags & DOM_LISTEN_ONCE) != 0 ? DOM_KIND_ONESHOT_ : 0),
                                             handler, userdata);
    if (id < 0) { return -1; }
    int jsId;
    unsafe { jsId = js_add_listener(handle, type, dom_len_(type), id, flags); }
    if (jsId < 0) {
        dom_unregister_callback(id);
        return -1;
    }
    unsafe { gDomCallbacks[id % DOM_MAX_CALLBACKS].extra = jsId; }
    return id;
}

int dom_on_selector_raw_(const &char selector, const &char type, void* handler, void* userdata) {
    int h = dom_query(selector);
    if (h < 0) { return -1; }
    return dom_add_listener_raw_(h, type, handler, userdata, 0);
}

void dom_off(int listenerId) {
    int slot = dom_slot_(listenerId);
    if (slot < 0) { return; }
    int jsId;
    unsafe { jsId = gDomCallbacks[slot].extra; }
    if (jsId >= 0) { unsafe { js_remove_listener(jsId); } }
    dom_unregister_callback(listenerId);
}

// ── Timers ─────────────────────────────────────────────────────────────────

int dom_set_timer_raw_(void* handler, void* userdata, int ms, int repeat) {
    int kind = DOM_CB_PLAIN_ | DOM_KIND_TIMER_ | (repeat ? 0 : DOM_KIND_ONESHOT_);
    int id = dom_register_callback_kind_raw_(kind, handler, userdata);
    if (id < 0) { return -1; }
    int jsId;
    unsafe { jsId = js_set_timer(id, ms, repeat); }
    unsafe { gDomCallbacks[id % DOM_MAX_CALLBACKS].extra = jsId; }
    return id;
}

void dom_clear_timer(int timerId) {
    int slot = dom_slot_(timerId);
    if (slot < 0) { return; }
    int jsId;
    unsafe { jsId = gDomCallbacks[slot].extra; }
    unsafe { js_clear_timer(jsId); }
    dom_unregister_callback(timerId);
}

int dom_request_animation_frame_raw_(void* handler, void* userdata) {
    int id = dom_register_callback_kind_raw_(DOM_CB_FRAME_ | DOM_KIND_ONESHOT_, handler, userdata);
    if (id < 0) { return -1; }
    int jsId;
    unsafe { jsId = js_request_frame(id); }
    unsafe { gDomCallbacks[id % DOM_MAX_CALLBACKS].extra = jsId; }
    return id;
}

void dom_cancel_animation_frame(int frameId) {
    int slot = dom_slot_(frameId);
    if (slot < 0) { return; }
    int jsId;
    unsafe { jsId = gDomCallbacks[slot].extra; }
    unsafe { js_cancel_frame(jsId); }
    dom_unregister_callback(frameId);
}

double dom_now() { unsafe { return js_now(); } }

// ── fetch() ────────────────────────────────────────────────────────────────

struct String DomFetchResponse::text() const { return dom_str_(DOM_STR_FETCH_, self.request, 0, (const char*)0, 0); }
struct String DomFetchResponse::status_text() const { return dom_str_(DOM_STR_FETCH_, self.request, 1, (const char*)0, 0); }
struct String DomFetchResponse::url() const { return dom_str_(DOM_STR_FETCH_, self.request, 2, (const char*)0, 0); }
struct String DomFetchResponse::header(const &char name) const {
    unsafe { return dom_str_(DOM_STR_FHEADER_, self.request, 0, name, dom_len_(name)); }
}

int dom_fetch_raw_(const &char method, const &char url, const ?&char headers, const ?&char body,
                   void* handler, void* userdata) {
    int id = dom_register_callback_kind_raw_(DOM_CB_FETCH_ | DOM_KIND_ONESHOT_, handler, userdata);
    if (id < 0) { return -1; }
    const char* hp = (const char*)0;
    int hl = -1;
    const char* bp = (const char*)0;
    int bl = -1;
    match (headers) { case some(h): { hp = h; hl = dom_len_(h); } case null: { } }
    match (body) { case some(b): { bp = b; bl = dom_len_(b); } case null: { } }
    int req;
    unsafe { req = js_fetch(id, method, dom_len_(method), url, dom_len_(url), hp, hl, bp, bl); }
    if (req < 0) {
        dom_unregister_callback(id);
        return -1;
    }
    unsafe { gDomCallbacks[id % DOM_MAX_CALLBACKS].extra = req; }
    return req;
}

// ── Storage ────────────────────────────────────────────────────────────────

int dom_storage_get(int kind, const &char key, &String out) {
    struct DomStrReq_ r;
    r.op = DOM_STR_STORAGE_;
    r.a = kind;
    r.b = 0;
    unsafe { r.s = key; }
    r.slen = dom_len_(key);
    return dom_str_into_(&r, out);
}

int dom_storage_set(int kind, const &char key, const &char value) {
    int rc;
    unsafe { rc = js_storage_set(kind, key, dom_len_(key), value, dom_len_(value)); }
    return rc == 0;
}

void dom_storage_remove(int kind, const &char key) { unsafe { js_storage_remove(kind, key, dom_len_(key)); } }
void dom_storage_clear(int kind) { unsafe { js_storage_clear(kind); } }
int  dom_storage_length(int kind) { unsafe { return js_storage_length(kind); } }
struct String dom_storage_key(int kind, int index) { return dom_str_(DOM_STR_SKEY_, kind, index, (const char*)0, 0); }

// ── Location / history ─────────────────────────────────────────────────────

struct String dom_location(int field) { return dom_str_(DOM_STR_LOCATION_, field, 0, (const char*)0, 0); }
struct String dom_location_path() { return dom_location(DOM_LOC_PATHNAME); }
void dom_history_push(const &char url) { unsafe { js_history(0, url, dom_len_(url)); } }
void dom_history_replace(const &char url) { unsafe { js_history(1, url, dom_len_(url)); } }
void dom_history_back() { unsafe { js_history(2, "", 0); } }
void dom_history_forward() { unsafe { js_history(3, "", 0); } }
void dom_navigate(const &char url) { unsafe { js_history(4, url, dom_len_(url)); } }

// ── Dispatch (called by the host) ──────────────────────────────────────────

typedef fn void(void* ud, &DomEvent ev) DomEventFn_;
typedef fn void(void* ud, double ts) DomFrameFn_;
typedef fn void(void* ud, &DomFetchResponse r) DomFetchFn_;

void dom_dispatch_event(int callbackId) {
    int slot = dom_slot_(callbackId);
    if (slot < 0) {
        // Ids below DOM_MAX_CALLBACKS come from glue written before
        // generations existed; they name slot+generation 1.
        if (callbackId >= 0 && callbackId < DOM_MAX_CALLBACKS) { slot = dom_slot_(callbackId + DOM_MAX_CALLBACKS); }
        if (slot < 0) { return; }
    }
    void* h;
    void* ud;
    int kind;
    unsafe {
        h = gDomCallbacks[slot].handler;
        ud = gDomCallbacks[slot].userdata;
        kind = gDomCallbacks[slot].kind;
    }
    if ((kind & DOM_KIND_ONESHOT_) != 0) { dom_unregister_callback(callbackId); }
    int base = kind & 15;
    if (base == DOM_CB_EVENT_) {
        struct DomEvent ev;
        ev.listener = callbackId;
        unsafe { ((DomEventFn_)h)(ud, &ev); }
    } else if (base == DOM_CB_PLAIN_) {
        unsafe { ((DomEventHandler)h)(ud); }
    }
}

void dom_dispatch_frame(int callbackId, double timestamp) {
    int slot = dom_slot_(callbackId);
    if (slot < 0) { return; }
    void* h;
    void* ud;
    unsafe {
        h = gDomCallbacks[slot].handler;
        ud = gDomCallbacks[slot].userdata;
    }
    dom_unregister_callback(callbackId);
    unsafe { ((DomFrameFn_)h)(ud, timestamp); }
}

void dom_dispatch_fetch(int callbackId, int requestId, int status) {
    int slot = dom_slot_(callbackId);
    if (slot < 0) {
        unsafe { js_fetch_done(requestId); }
        return;
    }
    void* h;
    void* ud;
    unsafe {
        h = gDomCallbacks[slot].handler;
        ud = gDomCallbacks[slot].userdata;
    }
    dom_unregister_callback(callbackId);
    struct DomFetchResponse r;
    r.request = requestId;
    r.status = status;
    r.ok = status >= 200 && status < 300;
    unsafe { ((DomFetchFn_)h)(ud, &r); }
    unsafe { js_fetch_done(requestId); }
}

} // namespace std
