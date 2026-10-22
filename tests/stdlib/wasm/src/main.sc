// std/wasm DOM interop test module: built for wasm32 by run.sh and driven
// by run.mjs under Node with fake_dom.mjs + std/wasm/safec_dom.js. Each
// exported step returns its failure count; failures are logged through
// dom_console (with the line) so run.mjs can show them.
#include <std/wasm/dom.h>
#include <std/wasm/hydrate.h>
#include <std/reactive/signal.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

static int g_fails = 0;

static void check_(int cond, int line) {
    if (cond) return;
    g_fails = g_fails + 1;
    char buf[24];
    std::wasm_itoa((long long)line, &buf[0], 24UL);
    struct String m = std::String::from("CHECK FAILED at main.sc:");
    m.push(&buf[0]);
    std::dom_console(DOM_LOG_ERROR, m.as_ptr());
    m.free();
}

static int str_is(struct String s, const &char want) {
    int r = s.eq_cstr(want);
    s.free();
    return r;
}

// ── elements, content, attributes, classes, style, properties ─────────────
__attribute__((export_name("t_elements"))) int t_elements() {
    g_fails = 0;
    int body = std::dom_body();
    check_((body >= 0) != 0, __LINE__);
    int list = std::dom_create_element("ul");
    std::dom_set_attr(list, "id", "list");
    std::dom_append(body, list);
    int i = 0;
    while (i < 3) {
        int li = std::dom_create_element("li");
        std::dom_class_add(li, "item");
        char buf[24];
        std::wasm_itoa((long long)i, &buf[0], 24UL);
        std::dom_set_text(li, &buf[0]);
        std::dom_append(list, li);
        std::dom_release(li);
        i = i + 1;
    }
    check_((std::dom_get_by_id("list") == list) != 0, __LINE__);
    struct Vec<int> items = std::Vec::new();
    check_((std::dom_query_all("#list .item", &items) == 3UL) != 0, __LINE__);
    check_((str_is(std::dom_get_text(*items.at(1UL)), "1")) != 0, __LINE__);
    check_((std::dom_parent(*items.at(0UL)) == list) != 0, __LINE__);
    check_((std::dom_next_sibling(*items.at(0UL)) == *items.at(1UL)) != 0, __LINE__);
    check_((std::dom_prev_sibling(*items.at(0UL)) == -1) != 0, __LINE__);
    check_((std::dom_first_child(list) == *items.at(0UL)) != 0, __LINE__);
    check_((std::dom_last_child(list) == *items.at(2UL)) != 0, __LINE__);
    struct Vec<int> kids = std::Vec::new();
    check_((std::dom_children(list, &kids) == 3UL) != 0, __LINE__);
    kids.free();

    // insert_before, remove
    int first = std::dom_create_element("li");
    std::dom_set_text(first, "first");
    std::dom_insert_before(list, first, *items.at(0UL));
    check_((str_is(std::dom_get_text(list), "first012")) != 0, __LINE__);
    std::dom_remove(*items.at(2UL));
    check_((str_is(std::dom_get_text(list), "first01")) != 0, __LINE__);
    items.free();

    // innerHTML
    int box = std::dom_create_element("div");
    std::dom_append(body, box);
    std::dom_set_html(box, "<p class=\"x\">a<b>b</b></p>");
    check_((std::dom_query_in(box, "p.x b") >= 0) != 0, __LINE__);
    check_((str_is(std::dom_get_html(box), "<p class=\"x\">a<b>b</b></p>")) != 0, __LINE__);
    check_((str_is(std::dom_get_text(box), "ab")) != 0, __LINE__);
    std::dom_clear(box);
    check_((str_is(std::dom_get_text(box), "")) != 0, __LINE__);

    // attributes
    std::dom_set_attr(box, "data-k", "v");
    check_((std::dom_has_attr(box, "data-k")) != 0, __LINE__);
    check_((str_is(std::dom_get_attr(box, "data-k"), "v")) != 0, __LINE__);
    std::dom_remove_attr(box, "data-k");
    check_((!std::dom_has_attr(box, "data-k")) != 0, __LINE__);
    check_((str_is(std::dom_get_attr(box, "nope"), "")) != 0, __LINE__);

    // classList
    std::dom_class_add(box, "a");
    check_((std::dom_class_contains(box, "a")) != 0, __LINE__);
    check_((std::dom_class_toggle(box, "b") == 1) != 0, __LINE__);
    check_((std::dom_class_toggle(box, "b") == 0) != 0, __LINE__);
    std::dom_class_remove(box, "a");
    check_((!std::dom_class_contains(box, "a")) != 0, __LINE__);

    // style
    std::dom_set_style(box, "background-color", "red");
    std::dom_set_visible(box, 0);

    // a long string (> the 256-byte first buffer)
    struct String big = std::String::repeat_str("0123456789", 100UL);
    std::dom_set_text(box, big.as_ptr());
    struct String back = std::dom_get_text(box);
    check_((back.length() == 1000UL) != 0, __LINE__);
    check_((back.eq(&big)) != 0, __LINE__);
    back.free();
    big.free();

    // properties, form controls
    int input = std::dom_create_element("input");
    std::dom_set_attr(input, "id", "name");
    std::dom_append(body, input);
    std::dom_set_value(input, "Ada");
    check_((str_is(std::dom_get_value(input), "Ada")) != 0, __LINE__);
    std::dom_set_checked(input, 1);
    check_((std::dom_get_checked(input)) != 0, __LINE__);
    std::dom_set_disabled(input, 1);
    check_((std::dom_get_prop_num(input, "disabled") == 1.0) != 0, __LINE__);
    std::dom_set_prop_num(input, "tabIndex", 3.0);
    check_((std::dom_get_prop_num(input, "tabIndex") == 3.0) != 0, __LINE__);
    char small[4];
    check_((std::dom_get_prop_buf(input, "value", &small[0], 4) == 3) != 0, __LINE__);
    check_((small[0] == 'A' && small[2] == 'a' && small[3] == (char)0) != 0, __LINE__);
    check_((std::dom_get_prop_buf(input, "missingProp", &small[0], 4) == -1) != 0, __LINE__);
    std::dom_focus(input);
    std::dom_set_prop(DOM_DOCUMENT, "title", "SafeC");
    check_((str_is(std::dom_get_prop(DOM_DOCUMENT, "title"), "SafeC")) != 0, __LINE__);
    check_((std::dom_query("#does-not-exist") == -1) != 0, __LINE__);

    std::dom_console(DOM_LOG_INFO, "elements done");
    return g_fails;
}

// ── events ─────────────────────────────────────────────────────────────────
struct Clicks {
    int count;
    int last_target;
    double x;
    int mods;
    int listener;
};
static struct Clicks g_clicks;
static struct String g_key;
static struct String g_typed;

static void on_click(?&Clicks c, &DomEvent ev) {
    match (c) {
        case some(k): {
            k.count = k.count + 1;
            k.last_target = ev.target();
            k.x = ev.client_x();
            k.mods = ev.modifiers();
            check_((str_is(ev.type(), "click")) != 0, __LINE__);
            ev.prevent_default();
        }
        case null: { }
    }
}

static void on_key(?&String out, &DomEvent ev) {
    match (out) {
        case some(s): {
            s.clear();
            struct String k = ev.key();
            s.push_str(&k);
            k.free();
            s.push_char('/');
            struct String c = ev.code();
            s.push_str(&c);
            c.free();
            if (ev.shift_key()) s.push("+shift");
        }
        case null: { }
    }
}

static void on_input(?&String out, &DomEvent ev) {
    match (out) {
        case some(s): {
            s.clear();
            struct String v = ev.target_value();
            s.push_str(&v);
            v.free();
        }
        case null: { }
    }
}

static int g_legacy = 0;
static void on_legacy(?&int n) { g_legacy = g_legacy + 1; }

__attribute__((export_name("t_events_setup"))) int t_events_setup() {
    g_fails = 0;
    g_clicks.count = 0;
    g_key = std::String::new();
    g_typed = std::String::new();
    int btn = std::dom_create_element("button");
    std::dom_set_attr(btn, "id", "btn");
    std::dom_append(std::dom_body(), btn);
    unsafe {
        g_clicks.listener = std::dom_on(btn, "click", on_click, (?&Clicks)&g_clicks);
        check_((g_clicks.listener >= 0) != 0, __LINE__);
        check_((std::dom_on(DOM_WINDOW, "keydown", on_key, (?&String)&g_key) >= 0) != 0, __LINE__);
        check_((std::dom_on_selector("#name", "input", on_input, (?&String)&g_typed) >= 0) != 0, __LINE__);
        check_((std::dom_on_click("#btn", on_legacy, (?&int)null) == btn) != 0, __LINE__);
    }
    return g_fails;
}

// JS clicked #btn twice (once with altKey and clientX 7), pressed a key on
// window, typed into #name.
__attribute__((export_name("t_events_check"))) int t_events_check() {
    g_fails = 0;
    check_((g_clicks.count == 2) != 0, __LINE__);
    check_((g_legacy == 2) != 0, __LINE__);
    check_((g_clicks.last_target == std::dom_get_by_id("btn")) != 0, __LINE__);
    check_((g_clicks.x == 7.0) != 0, __LINE__);
    check_((g_clicks.mods == DOM_MOD_ALT) != 0, __LINE__);
    check_((g_key.eq_cstr("A/KeyA+shift")) != 0, __LINE__);
    check_((g_typed.eq_cstr("typed text")) != 0, __LINE__);
    std::dom_off(g_clicks.listener);
    return g_fails;
}

__attribute__((export_name("clicks"))) int clicks() { return g_clicks.count; }

// ── timers, frames, fetch, storage, history ────────────────────────────────
struct Async {
    int timeouts;
    int ticks;
    int interval;
    double frame_ts;
    int cancelled_ran;
    int fetch_status;
    struct String fetch_body;
    struct String fetch_header;
    int fetch_fail_status;
    int pops;
    struct String pop_state;
};
static struct Async g_async;

static void on_timeout(?&Async a) { match (a) { case some(s): { s.timeouts = s.timeouts + 1; } case null: { } } }
static void on_cancelled(?&Async a) { match (a) { case some(s): { s.cancelled_ran = 1; } case null: { } } }
static void on_tick(?&Async a) {
    match (a) {
        case some(s): {
            s.ticks = s.ticks + 1;
            if (s.ticks == 3) std::dom_clear_timer(s.interval);
        }
        case null: { }
    }
}
static void on_frame(?&Async a, double ts) { match (a) { case some(s): { s.frame_ts = ts; } case null: { } } }
static void on_fetch(?&Async a, &DomFetchResponse r) {
    match (a) {
        case some(s): {
            s.fetch_status = r.status;
            check_((r.ok) != 0, __LINE__);
            struct String t = r.text();
            s.fetch_body.push_str(&t);
            t.free();
            struct String h = r.header("Content-Type");
            s.fetch_header.push_str(&h);
            h.free();
            check_((str_is(r.url(), "http://example.test/api/data")) != 0, __LINE__);
        }
        case null: { }
    }
}
static void on_fetch_fail(?&Async a, &DomFetchResponse r) {
    match (a) { case some(s): { s.fetch_fail_status = r.status + 1000; check_((!r.ok) != 0, __LINE__); } case null: { } }
}
static void on_pop(?&Async a, &DomEvent ev) {
    match (a) {
        case some(s): {
            s.pops = s.pops + 1;
            s.pop_state.clear();
            struct String st = ev.state_json();
            s.pop_state.push_str(&st);
            st.free();
        }
        case null: { }
    }
}

__attribute__((export_name("t_async_setup"))) int t_async_setup() {
    g_fails = 0;
    g_async.timeouts = 0;
    g_async.ticks = 0;
    g_async.frame_ts = 0.0;
    g_async.cancelled_ran = 0;
    g_async.fetch_status = 0;
    g_async.fetch_body = std::String::new();
    g_async.fetch_header = std::String::new();
    g_async.fetch_fail_status = 0;
    g_async.pops = 0;
    g_async.pop_state = std::String::new();
    unsafe {
        ?&Async a = (?&Async)&g_async;
        check_((std::dom_set_timeout(on_timeout, a, 5) >= 0) != 0, __LINE__);
        int t = std::dom_set_timeout(on_cancelled, a, 5);
        std::dom_clear_timer(t);
        g_async.interval = std::dom_set_interval(on_tick, a, 2);
        check_((std::dom_request_animation_frame(on_frame, a) >= 0) != 0, __LINE__);
        int f = std::dom_request_animation_frame(on_frame, a);
        std::dom_cancel_animation_frame(f);
        check_((std::dom_fetch("POST", "/api/data", "Content-Type: application/json\nX-A: 1", "{\"q\":1}", on_fetch, a) >= 0) != 0, __LINE__);
        check_((std::dom_fetch_get("/missing", on_fetch_fail, a) >= 0) != 0, __LINE__);
        check_((std::dom_on(DOM_WINDOW, "popstate", on_pop, a) >= 0) != 0, __LINE__);
    }
    check_((std::dom_now() == 1234.5) != 0, __LINE__);

    // storage (synchronous)
    check_((std::dom_storage_set(DOM_STORAGE_LOCAL, "theme", "dark")) != 0, __LINE__);
    struct String v = std::String::new();
    check_((std::dom_storage_get(DOM_STORAGE_LOCAL, "theme", &v)) != 0, __LINE__);
    check_((v.eq_cstr("dark")) != 0, __LINE__);
    v.clear();
    check_((!std::dom_storage_get(DOM_STORAGE_LOCAL, "absent", &v)) != 0, __LINE__);
    check_((std::dom_storage_length(DOM_STORAGE_LOCAL) == 1) != 0, __LINE__);
    check_((str_is(std::dom_storage_key(DOM_STORAGE_LOCAL, 0), "theme")) != 0, __LINE__);
    struct String huge = std::String::repeat_str("x", 2000UL);
    check_((!std::dom_storage_set(DOM_STORAGE_SESSION, "big", huge.as_ptr())) != 0, __LINE__);   // fake quota
    huge.free();
    std::dom_storage_set(DOM_STORAGE_SESSION, "s", "1");
    std::dom_storage_remove(DOM_STORAGE_SESSION, "s");
    check_((std::dom_storage_length(DOM_STORAGE_SESSION) == 0) != 0, __LINE__);
    std::dom_storage_set(DOM_STORAGE_SESSION, "a", "1");
    std::dom_storage_clear(DOM_STORAGE_SESSION);
    check_((std::dom_storage_length(DOM_STORAGE_SESSION) == 0) != 0, __LINE__);
    v.free();

    // history + location
    std::dom_history_push("/docs/intro?x=1#top");
    check_((str_is(std::dom_location_path(), "/docs/intro")) != 0, __LINE__);
    check_((str_is(std::dom_location(DOM_LOC_SEARCH), "?x=1")) != 0, __LINE__);
    check_((str_is(std::dom_location(DOM_LOC_HASH), "#top")) != 0, __LINE__);
    check_((str_is(std::dom_location(DOM_LOC_HREF), "http://example.test/docs/intro?x=1#top")) != 0, __LINE__);
    check_((str_is(std::dom_location(DOM_LOC_ORIGIN), "http://example.test")) != 0, __LINE__);
    std::dom_history_push("/docs/next");
    std::dom_history_replace("/docs/next2");
    std::dom_history_back();          // fires popstate synchronously in the fake
    check_((str_is(std::dom_location_path(), "/docs/intro")) != 0, __LINE__);
    std::dom_navigate("/elsewhere");
    return g_fails;
}

__attribute__((export_name("t_async_check"))) int t_async_check() {
    g_fails = 0;
    check_((g_async.timeouts == 1) != 0, __LINE__);
    check_((g_async.cancelled_ran == 0) != 0, __LINE__);
    check_((g_async.ticks == 3) != 0, __LINE__);
    check_((g_async.frame_ts == 16.5) != 0, __LINE__);
    check_((g_async.fetch_status == 201) != 0, __LINE__);
    check_((g_async.fetch_body.eq_cstr("echo:{\"q\":1}|application/json|1")) != 0, __LINE__);
    check_((g_async.fetch_header.eq_cstr("text/plain")) != 0, __LINE__);
    check_((g_async.fetch_fail_status == 1000) != 0, __LINE__);
    check_((g_async.pops == 1) != 0, __LINE__);
    check_((g_async.pop_state.eq_cstr("null")) != 0, __LINE__);
    return g_fails;
}

// ── signals driving the DOM ────────────────────────────────────────────────
static struct Signal<int> g_count;
static struct Signal<int> g_flag;
static struct Signal<String> g_name;
static struct Signal<String> g_title;

static void fmt_count(&String out, const &int v) {
    out.push("count=");
    out.push_int((long long)*v);
}

__attribute__((export_name("t_hydrate"))) int t_hydrate() {
    g_fails = 0;
    int body = std::dom_body();
    int count = std::dom_create_element("span");
    int label = std::dom_create_element("span");
    int panel = std::dom_create_element("div");
    int title = std::dom_create_element("h1");
    int field = std::dom_create_element("input");
    std::dom_set_attr(field, "id", "field");
    int box = std::dom_create_element("input");
    std::dom_set_attr(box, "id", "box");
    std::dom_append(body, count);
    std::dom_append(body, label);
    std::dom_append(body, panel);
    std::dom_append(body, title);
    std::dom_append(body, field);
    std::dom_append(body, box);

    g_count = std::Signal::new(5);
    g_flag = std::Signal::new(0);
    g_name = std::Signal::new(std::String::from("Ada"));
    g_title = std::Signal::new(std::String::from("Hello"));
    std::hydrate_bind_text_with(&g_count, count, fmt_count);
    check_((str_is(std::dom_get_text(count), "count=5")) != 0, __LINE__);
    g_count.set(6);
    check_((str_is(std::dom_get_text(count), "count=6")) != 0, __LINE__);

    std::hydrate_bind_text_int_handle(&g_count, label);
    g_count.set(7);
    check_((str_is(std::dom_get_text(label), "7")) != 0, __LINE__);

    std::hydrate_bind_class(&g_flag, panel, "on");
    std::hydrate_bind_visible(&g_flag, panel);
    check_((!std::dom_class_contains(panel, "on")) != 0, __LINE__);
    g_flag.set(1);
    check_((std::dom_class_contains(panel, "on")) != 0, __LINE__);

    std::hydrate_bind_text(&g_title, title);
    std::hydrate_bind_attr(&g_title, title, "data-title");
    check_((str_is(std::dom_get_text(title), "Hello")) != 0, __LINE__);
    check_((std::hydrate_bind_value(&g_name, field) >= 0) != 0, __LINE__);
    check_((std::hydrate_bind_checked(&g_flag, box) >= 0) != 0, __LINE__);
    check_((str_is(std::dom_get_value(field), "Ada")) != 0, __LINE__);
    check_((std::dom_get_checked(box)) != 0, __LINE__);
    return g_fails;
}

// JS typed "Grace" into #field and unchecked #box.
__attribute__((export_name("t_hydrate_check"))) int t_hydrate_check() {
    g_fails = 0;
    check_((g_name.get_ref().eq_cstr("Grace")) != 0, __LINE__);
    check_((g_flag.get() == 0) != 0, __LINE__);
    int panel = std::dom_query("div");
    check_((panel >= 0) != 0, __LINE__);
    struct String t = std::String::from("Bye");
    &String cur = g_title.get_ref();
    cur.free();
    *cur = t;
    g_title.notify();
    check_((str_is(std::dom_get_attr(std::dom_query("h1"), "data-title"), "Bye")) != 0, __LINE__);
    return g_fails;
}
