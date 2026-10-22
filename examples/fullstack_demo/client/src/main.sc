// fullstack_demo — WASM client: hydrates the /counter page (see
// src/pages.scx's render_counter()). A package of its own, built for
// wasm32 and copied next to the server's static files:
//
//   cd client && safeguard build --target wasm32
//   cp build/counter_client ../static/counter_client.wasm
//
// The page loads it through static/app.js, which uses std's JS glue
// (static/safec_dom.js, a copy of <SafeC>/std/wasm/safec_dom.js) to
// provide every std/wasm/dom.h import.
//
// What it shows: a Signal<int> bound to the DOM (std/wasm/hydrate.h),
// click and keyboard listeners (dom_on), localStorage persistence, a
// fetch() to the server's JSON API, and a two-way bound text input.
#include <std/wasm/wasm_rt.h>
#include <std/reactive/signal.h>
#include <std/wasm/dom.h>
#include <std/wasm/hydrate.h>

static struct Signal<int> gCounter;
static struct Signal<String> gName;

static void save(?&Signal<int> s) {
    match (s) {
        case some(c): {
            char buf[24];
            std::wasm_itoa((long long)c.get(), &buf[0], 24UL);
            std::dom_storage_set(DOM_STORAGE_LOCAL, "fullstack_demo.count", &buf[0]);
        }
        case null: { }
    }
}

static void on_increment(?&Signal<int> s, &DomEvent ev) {
    match (s) { case some(c): { c.set(c.get() + 1); } case null: { } }
}

static void on_decrement(?&Signal<int> s, &DomEvent ev) {
    match (s) { case some(c): { c.set(c.get() - 1); } case null: { } }
}

// '+' / '-' keys anywhere on the page.
static void on_key(?&Signal<int> s, &DomEvent ev) {
    struct String k = ev.key();
    match (s) {
        case some(c): {
            if (k.eq_cstr("+") || k.eq_cstr("ArrowUp")) { c.set(c.get() + 1); ev.prevent_default(); }
            if (k.eq_cstr("-") || k.eq_cstr("ArrowDown")) { c.set(c.get() - 1); ev.prevent_default(); }
        }
        case null: { }
    }
    k.free();
}

static void on_status(?&int unused, &DomFetchResponse r) {
    int el = std::dom_query("#server-status");
    if (el < 0) { return; }
    if (r.ok) {
        struct String body = r.text();
        std::dom_set_text(el, body.as_ptr());
        body.free();
    } else {
        std::dom_set_text(el, "(server unreachable)");
    }
    std::dom_release(el);
}

static void fmt_greeting(&String out, const &String name) {
    out.push("Hello, ");
    if (name.is_empty()) { out.push("stranger"); } else { out.push_str(name); }
    out.push_char('!');
}

// Exported: app.js calls this once the module is instantiated.
__attribute__((export_name)) int hydrate_init() {
    int start = 0;
    struct String saved = std::String::new();
    if (std::dom_storage_get(DOM_STORAGE_LOCAL, "fullstack_demo.count", &saved)) {
        int ok = 0;
        start = (int)saved.parse_int(&ok);
        if (!ok) { start = 0; }
    }
    saved.free();
    gCounter = std::Signal::new(start);

    int countHandle = std::dom_get_element("#count");
    char buf[24];
    std::wasm_itoa((long long)start, &buf[0], 24UL);
    std::dom_set_text(countHandle, &buf[0]);
    std::hydrate_bind_text_int_handle(&gCounter, countHandle);
    unsafe {
        std::signal_subscribe(&gCounter, save, (?&Signal<int>)&gCounter);
        std::dom_on_selector("#increment", "click", on_increment, (?&Signal<int>)&gCounter);
        std::dom_on_selector("#decrement", "click", on_decrement, (?&Signal<int>)&gCounter);
        std::dom_on(DOM_WINDOW, "keydown", on_key, (?&Signal<int>)&gCounter);
    }

    gName = std::Signal::new(std::String::new());
    int input = std::dom_query("#name");
    int greeting = std::dom_query("#greeting");
    if (input >= 0 && greeting >= 0) {
        std::hydrate_bind_value(&gName, input);
        std::hydrate_bind_text_with(&gName, greeting, fmt_greeting);
    }

    std::dom_fetch_get("/api/status", on_status, null);
    std::dom_console(DOM_LOG_INFO, "counter_client.wasm hydrated");
    return countHandle;
}
