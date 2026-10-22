// std::gui GuiApp: several windows from one loop — per-window event
// routing, widget roots, draw/frame/event callbacks, timers on the
// headless backend's virtual clock, close handling (and refusing a
// close), quit(), the window registry, and the clipboard.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/collections/string.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_headless.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>

extern void* memset(void* p, int v, unsigned long n);

struct Counts {
    int events;
    int keys;
    int closes;
    int draws;
    int frames;
    int clicks;
    int timeouts;
    int ticks;
    double lastFrame;
    unsigned int intervalId;
    int refuseClose;
};

static int on_event(&GuiApp app, &GuiWindow win, &GuiEvent ev, ?&Counts oc) {
    match (oc) {
        case some(c): {
            c.events = c.events + 1;
            if (ev.kind == GUI_EVENT_KEY_DOWN) { c.keys = c.keys + 1; }
            if (ev.kind == GUI_EVENT_CLOSE) {
                c.closes = c.closes + 1;
                if (c.refuseClose) { ev.prevent_default(); }
            }
        }
        case null: { }
    }
    return 0;
}

static void on_draw(&GuiApp app, &GuiWindow win, &GuiCanvas cv, ?&Counts oc) {
    match (oc) { case some(c): { c.draws = c.draws + 1; } case null: { } }
    cv.set_fill_color(GuiColor::hex(0x00FF00U));
    cv.fill_rect(0.0f, 0.0f, 4.0f, 4.0f);
}

static int on_frame(&GuiApp app, &GuiWindow win, double t, ?&Counts oc) {
    match (oc) {
        case some(c): {
            c.frames = c.frames + 1;
            c.lastFrame = t;
            return c.frames < 5 ? 1 : 0;              // stop after five frames
        }
        case null: { }
    }
    return 0;
}

static void on_timeout(&GuiApp app, unsigned int id, ?&Counts oc) {
    match (oc) { case some(c): { c.timeouts = c.timeouts + 1; } case null: { } }
}

static void on_tick(&GuiApp app, unsigned int id, ?&Counts oc) {
    match (oc) {
        case some(c): {
            c.ticks = c.ticks + 1;
            if (c.ticks == 3) { app.cancel_timer(id); }
        }
        case null: { }
    }
}

static void on_quit(&GuiApp app, unsigned int id, ?&Counts oc) { app.quit(7); }

static void on_click(&Widget w, ?&Counts oc) {
    match (oc) { case some(c): { c.clicks = c.clicks + 1; } case null: { } }
}

static struct Counts counts() {
    struct Counts c;
    unsafe { memset((void*)&c, 0, sizeof(struct Counts)); }
    return c;
}

static &GuiWindow must(?&GuiWindow ow) {
    match (ow) { case some(w): { return w; } case null: { } }
    ASSERT_TRUE(0);
    unsafe { return (struct GuiWindow*)0; }
}

void t_windows_and_routing() {
    struct GuiApp app = GuiApp::new();
    struct Counts ca = counts();
    struct Counts cb = counts();
    &GuiWindow a = must(app.open_window("A", 200, 100));
    struct GuiWindowOptions o = GuiWindowOptions::new("B", 300, 150);
    o.x = 40; o.y = 50;
    &GuiWindow b = must(app.open(&o));
    ASSERT_EQ(app.window_count(), 2UL);
    ASSERT_TRUE(a.id != b.id);
    ASSERT_EQ(b.pos_x(), 40); ASSERT_EQ(b.pos_y(), 50);
    match (app.find_window(b.id)) {
        case some(w): { ASSERT_EQ(w.id, b.id); }
        case null: { ASSERT_TRUE(0); }
    }
    ASSERT_TRUE(app.find_window(99999U).is_null());
    app.on_event(a, on_event, &ca);
    app.on_event(b, on_event, &cb);
    app.on_draw(b, on_draw, &cb);

    // Both are painted and presented on the first iteration.
    ASSERT_EQ(app.pump(0), 1);
    ASSERT_EQ(cb.draws, 1);
    ASSERT_TRUE(gui_headless_present_count(a) >= 1UL);
    ASSERT_TRUE(gui_headless_present_count(b) >= 1UL);
    ASSERT_EQ(b.get_pixel(1, 1), 0x00FF00FFU);
    ASSERT_EQ(a.get_pixel(1, 1), 0U);                 // no root, no draw callback: left alone
    // Nothing changed: no repaint.
    unsigned long pa = gui_headless_present_count(a);
    app.pump(0);
    ASSERT_EQ(cb.draws, 1);
    ASSERT_EQ(gui_headless_present_count(a), pa);
    app.redraw(b);
    app.pump(0);
    ASSERT_EQ(cb.draws, 2);

    // Events go to the window they're for.
    gui_headless_key(b, GUI_KEY_ENTER, 0, 0U);
    gui_headless_key(b, GUI_KEY_SPACE, 0, 32U);
    gui_headless_mouse_move(a, 10.0, 10.0);
    app.pump(0);
    ASSERT_EQ(cb.keys, 2);
    ASSERT_EQ(ca.keys, 0);
    ASSERT_TRUE(ca.events >= 1);

    // Resizing reallocates the buffer and repaints.
    int drawsBefore = cb.draws;
    gui_headless_resize(b, 320, 160);
    app.pump(0);
    ASSERT_EQ(b.width, 320); ASSERT_EQ(b.height, 160);
    ASSERT_EQ(cb.draws, drawsBefore + 1);

    // Closing: refused once, then accepted.
    cb.refuseClose = 1;
    gui_headless_close(b);
    ASSERT_EQ(app.pump(0), 1);
    ASSERT_EQ(cb.closes, 1);
    ASSERT_EQ(app.window_count(), 2UL);
    cb.refuseClose = 0;
    gui_headless_close(b);
    ASSERT_EQ(app.pump(0), 1);
    ASSERT_EQ(cb.closes, 2);
    ASSERT_EQ(app.window_count(), 1UL);
    ASSERT_EQ(gui_backend_window_count(), 1UL);
    // The last window closing ends the loop.
    gui_headless_close(a);
    ASSERT_EQ(app.pump(0), 0);
    ASSERT_EQ(app.window_count(), 0UL);
    app.free();
}

void t_widget_roots() {
    struct GuiApp app = GuiApp::new();
    struct Counts c1 = counts();
    struct Counts c2 = counts();
    &GuiWindow w1 = must(app.open_window("one", 200, 120));
    &GuiWindow w2 = must(app.open_window("two", 200, 120));
    &Widget r1 = Widget::vstack().set_padding(10);
    &Widget b1 = Widget::button("First").on_click(on_click, &c1);
    r1.add_child(b1);
    &Widget r2 = Widget::vstack().set_padding(10);
    &Widget b2 = Widget::button("Second").on_click(on_click, &c2);
    r2.add_child(b2);
    &GuiUi u1 = app.set_root(w1, r1);
    app.set_root(w2, r2);
    match (app.ui_of(w1)) {
        case some(u): { unsafe { ASSERT_TRUE((void*)u == (void*)u1); } }
        case null: { ASSERT_TRUE(0); }
    }
    app.pump(0);
    ASSERT_TRUE(b1.w > 0 && b2.w > 0);
    // Same position in both windows: only the window clicked reacts.
    gui_headless_click(w2, (double)b2.x + 5.0, (double)b2.y + 5.0);
    app.pump(0);
    ASSERT_EQ(c1.clicks, 0);
    ASSERT_EQ(c2.clicks, 1);
    gui_headless_click(w1, (double)b1.x + 5.0, (double)b1.y + 5.0);
    app.pump(0);
    ASSERT_EQ(c1.clicks, 1);
    ASSERT_EQ(c2.clicks, 1);
    // Focus is per window.
    gui_headless_key(w2, GUI_KEY_TAB, 0, 0U);
    app.pump(0);
    ASSERT_EQ(b1.focused, 1);
    ASSERT_EQ(b2.focused, 1);
    // A widget root repaints when its state changes.
    unsigned long p = gui_headless_present_count(w1);
    b1.set_text("Changed");
    app.pump(0);
    ASSERT_TRUE(gui_headless_present_count(w1) > p);
    // Closing a window frees its tree (no leak checks here; it must not crash).
    app.close_window(w2);
    ASSERT_EQ(app.window_count(), 1UL);
    app.pump(0);
    app.free();
    ASSERT_EQ(gui_backend_window_count(), 0UL);
}

void t_timers_and_frames() {
    struct GuiApp app = GuiApp::new();
    struct Counts c = counts();
    &GuiWindow w = must(app.open_window("timers", 100, 100));
    double t0 = app.now_ms();
    unsigned int id1 = app.set_timeout(100.0, on_timeout, &c);
    unsigned int id2 = app.set_interval(30.0, on_tick, &c);
    unsigned int id3 = app.set_timeout(50.0, on_timeout, &c);
    ASSERT_TRUE(id1 != 0U && id2 != 0U && id3 != 0U && id1 != id2 && id2 != id3);
    app.cancel_timer(id3);
    // pump(-1) waits (in virtual time) for the next timer.
    app.pump(0);
    ASSERT_EQ(c.ticks, 0);
    app.pump(-1);
    ASSERT_EQ(c.ticks, 1);
    ASSERT_TRUE(app.now_ms() - t0 >= 30.0 && app.now_ms() - t0 < 31.5);
    while (app.now_ms() - t0 < 200.0) { app.pump(10); }  // pump(10): up to 10 ms
    ASSERT_EQ(c.timeouts, 1);                         // the cancelled one never ran
    ASSERT_EQ(c.ticks, 3);                            // cancelled itself on the third tick
    // Frame callbacks: 60 Hz while they return 1.
    app.set_frame_interval(20.0);
    ASSERT_TRUE(app.frame_interval() == 20.0);
    app.on_frame(w, on_frame, &c);
    double f0 = app.now_ms();
    int guard = 0;
    while (c.frames < 5 && guard < 100) { app.pump(-1); guard = guard + 1; }
    ASSERT_EQ(c.frames, 5);
    ASSERT_TRUE(c.lastFrame - f0 >= 60.0 && c.lastFrame - f0 <= 101.0);
    // It returned 0: no more frames.
    double t1 = app.now_ms();
    app.set_timeout(500.0, on_timeout, &c);
    app.pump(-1);
    ASSERT_EQ(c.frames, 5);
    ASSERT_TRUE(app.now_ms() - t1 >= 499.0);
    // run() returns quit()'s code.
    app.set_timeout(10.0, on_quit, &c);
    ASSERT_EQ(app.run(), 7);
    app.free();
}

void t_window_state() {
    struct GuiApp app = GuiApp::new();
    struct GuiWindowOptions o = GuiWindowOptions::new("state", 320, 200);
    o.minWidth = 100; o.minHeight = 80;
    &GuiWindow w = must(app.open(&o));
    w.set_title("renamed");
    match (gui_window_state(w)) {
        case some(st): { ASSERT_TRUE(st.title.eq_cstr("renamed")); }
        case null: { ASSERT_TRUE(0); }
    }
    w.set_size(400, 300);
    app.pump(0);
    ASSERT_EQ(w.width, 400); ASSERT_EQ(w.height, 300);
    w.set_size(10, 10);                               // the minimum size holds
    app.pump(0);
    ASSERT_EQ(w.width, 100); ASSERT_EQ(w.height, 80);
    w.set_position(70, 80);
    ASSERT_EQ(w.pos_x(), 70);
    w.hide();
    ASSERT_EQ(w.is_visible(), 0);
    w.show();
    ASSERT_EQ(w.is_visible(), 1);
    w.focus();
    ASSERT_EQ(w.is_focused(), 1);
    w.set_cursor(GUI_CURSOR_HAND);
    ASSERT_EQ(gui_headless_cursor(w), GUI_CURSOR_HAND);
    // HiDPI: logical vs buffer sizes.
    gui_headless_set_scale(1.5);
    &GuiWindow h = must(app.open_window("hidpi", 200, 100));
    ASSERT_EQ(h.width, 300); ASSERT_EQ(h.height, 150);
    ASSERT_EQ(h.logical_width(), 200);
    ASSERT_TRUE(h.scale == 1.5f);
    gui_headless_set_scale(1.0);
    // The clipboard.
    ASSERT_TRUE(gui_clipboard_set_text("copied \xE2\x9C\x93"));
    struct String s = String::new();
    ASSERT_TRUE(gui_clipboard_get_text(&s));
    ASSERT_TRUE(s.eq_cstr("copied \xE2\x9C\x93"));
    s.free();
    w.request_close();
    app.pump(0);
    ASSERT_EQ(app.window_count(), 1UL);
    app.free();
}

int main() {
    gui_use_headless();
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("windows, event routing, repaint, close", t_windows_and_routing);
        t.add("widget roots in two windows", t_widget_roots);
        t.add("timers and frame callbacks", t_timers_and_frames);
        t.add("window state, HiDPI, clipboard", t_window_state);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
