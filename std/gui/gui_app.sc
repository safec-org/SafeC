// SafeC Standard Library — GUI: the application object and its event loop
// (see gui_app.h).
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/mem.h>
#include <std/collections/vec.h>
extern void* memset(void* ptr, int val, size_t n);

namespace std {

struct GuiApp GuiApp::new() {
    struct GuiApp a;
    a.windows = Vec::new();
    a.timers = Vec::new();
    a.nextTimer = 1U;
    a.quitting = 0;
    a.exitCode = 0;
    a.frameMs = 1000.0 / 60.0;
    a.lastFrame = -1000000000.0;
    return a;
}

double GuiApp::now_ms() const { return gui_time_ms(); }
void GuiApp::set_frame_interval(double ms) { self.frameMs = ms > 1.0 ? ms : 1.0; }
double GuiApp::frame_interval() const { return self.frameMs; }
void GuiApp::quit(int exitCode) { self.quitting = 1; self.exitCode = exitCode; }

static ?&GuiAppWindow __gui_app_rec(const &GuiApp app, const &GuiWindow win) {
    unsigned long i = 0UL;
    while (i < app.windows.len) {
        &GuiAppWindow r = *app.windows.at(i);
        unsafe { if ((const void*)r.win == (const void*)win) { return r; } }
        i = i + 1UL;
    }
    return null;
}

?&GuiWindow GuiApp::open(const &GuiWindowOptions opts) {
    struct GuiWindow w = GuiWindow::create_with(opts);
    if (w.platform.is_null()) {
        w.destroy();
        return null;
    }
    &GuiWindow hw;
    &GuiAppWindow rec;
    unsafe {
        hw = (struct GuiWindow*)alloc_raw_(sizeof(struct GuiWindow));
        rec = (struct GuiAppWindow*)alloc_zeroed_raw_(sizeof(struct GuiAppWindow));
    }
    *hw = w;
    rec.win = hw;
    rec.ui = null;
    unsafe {
        rec.onEvent = null; rec.onEventData = null;
        rec.onDraw = null; rec.onDrawData = null;
        rec.onFrame = null; rec.onFrameData = null;
    }
    rec.canvasReady = 0;
    rec.dirty = 1;
    rec.closing = 0;
    self.windows.push(rec);
    return hw;
}

?&GuiWindow GuiApp::open_window(const &char title, int width, int height) {
    struct GuiWindowOptions o = GuiWindowOptions::new(title, width, height);
    return self.open(&o);
}

void GuiApp::close_window(&GuiWindow win) {
    unsigned long i = 0UL;
    while (i < self.windows.len) {
        &GuiAppWindow r = *self.windows.at(i);
        int same = 0;
        unsafe { same = ((void*)r.win == (void*)win) ? 1 : 0; }
        if (same) {
            self.windows.remove(i);
            match (r.ui) { case some(ui): { ui.free(); } case null: { } }
            unsafe {
                // The record (raw heap, freed below) is taken apart in place.
                if (r.canvasReady) { r.canvas.free(); }
                r.win.destroy();
                dealloc((?&heap GuiWindow)r.win);
                dealloc((?&heap GuiAppWindow)r);
            }
            return;
        }
        i = i + 1UL;
    }
}

unsigned long GuiApp::window_count() const { return self.windows.len; }

?&GuiWindow GuiApp::window_at(unsigned long i) const {
    if (i >= self.windows.len) { return null; }
    &GuiAppWindow r = *self.windows.at(i);
    return r.win;
}

?&GuiWindow GuiApp::find_window(unsigned int id) const {
    unsigned long i = 0UL;
    while (i < self.windows.len) {
        &GuiAppWindow r = *self.windows.at(i);
        if (r.win.id == id) { return r.win; }
        i = i + 1UL;
    }
    return null;
}

&GuiUi GuiApp::set_root(&GuiWindow win, &Widget root) {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    &GuiUi ui = GuiUi::new(root);
    ui.theme = GuiTheme::system();
    ui.attach(win);
    match (orec) {
        case some(rec): {
            match (rec.ui) { case some(old): { old.free(); } case null: { } }
            rec.ui = ui;
            rec.dirty = 1;
        }
        case null: { }
    }
    return ui;
}

?&GuiUi GuiApp::ui_of(const &GuiWindow win) const {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    match (orec) { case some(rec): { return rec.ui; } case null: { } }
    return null;
}

void GuiApp::redraw(&GuiWindow win) {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    match (orec) { case some(rec): { rec.dirty = 1; } case null: { } }
}

void GuiApp::on_event_raw_(&GuiWindow win, fn int(&GuiApp, &GuiWindow, &GuiEvent, ?&Erased) cb, ?&Erased data) {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    match (orec) { case some(rec): { unsafe { rec.onEvent = cb; rec.onEventData = data; } } case null: { } }
}

void GuiApp::on_draw_raw_(&GuiWindow win, fn void(&GuiApp, &GuiWindow, &GuiCanvas, ?&Erased) cb, ?&Erased data) {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    match (orec) { case some(rec): { unsafe { rec.onDraw = cb; rec.onDrawData = data; } rec.dirty = 1; } case null: { } }
}

void GuiApp::on_frame_raw_(&GuiWindow win, fn int(&GuiApp, &GuiWindow, double, ?&Erased) cb, ?&Erased data) {
    ?&GuiAppWindow orec = __gui_app_rec(self, win);
    match (orec) { case some(rec): { unsafe { rec.onFrame = cb; rec.onFrameData = data; } } case null: { } }
}

unsigned int GuiApp::add_timer_raw_(double ms, int repeat, fn void(&GuiApp, unsigned int, ?&Erased) cb, ?&Erased data) {
    struct GuiTimer t;
    t.id = self.nextTimer;
    self.nextTimer = self.nextTimer + 1U;
    if (self.nextTimer == 0U) { self.nextTimer = 1U; }
    if (ms < 0.0) { ms = 0.0; }
    t.due = self.now_ms() + ms;
    t.interval = repeat ? (ms > 0.5 ? ms : 0.5) : 0.0;
    t.cb = cb; t.data = data;
    t.dead = 0;
    self.timers.push(t);
    return t.id;
}

void GuiApp::cancel_timer(unsigned int id) {
    unsigned long i = 0UL;
    while (i < self.timers.len) {
        &GuiTimer t = self.timers.at(i);
        if (t.id == id) { t.dead = 1; }
        i = i + 1UL;
    }
}

static int __gui_app_animating(&GuiApp app) {
    unsigned long i = 0UL;
    while (i < app.windows.len) {
        &GuiAppWindow r = *app.windows.at(i);
        if (!r.onFrame.is_null()) { return 1; }
        match (r.ui) {
            case some(ui): { if (ui.animating() || !ui.tipFor.is_null()) { return 1; } }
            case null: { }
        }
        i = i + 1UL;
    }
    return 0;
}

static int __gui_app_dirty(&GuiApp app) {
    unsigned long i = 0UL;
    while (i < app.windows.len) {
        &GuiAppWindow r = *app.windows.at(i);
        if (r.dirty) { return 1; }
        match (r.ui) { case some(ui): { if (ui.needs_redraw()) { return 1; } } case null: { } }
        i = i + 1UL;
    }
    return 0;
}

int GuiApp::pump(int timeoutMs) {
    if (self.quitting || self.windows.len == 0UL) { return 0; }
    double now = self.now_ms();
    // How long the platform may wait for events before something is due.
    double wait = timeoutMs < 0 ? -1.0 : (double)timeoutMs;
    if (__gui_app_dirty(self) || gui_backend_has_events()) { wait = 0.0; }
    unsigned long ti = 0UL;
    while (ti < self.timers.len) {
        &GuiTimer t = self.timers.at(ti);
        if (!t.dead) {
            double d = t.due - now;
            if (d < 0.0) { d = 0.0; }
            if (wait < 0.0 || d < wait) { wait = d; }
        }
        ti = ti + 1UL;
    }
    if (__gui_app_animating(self)) {
        double d = self.lastFrame + self.frameMs - now;
        if (d < 0.0) { d = 0.0; }
        if (wait < 0.0 || d < wait) { wait = d; }
    }
    int waitMs = wait < 0.0 ? -1 : (int)(wait + 0.999);
    &GuiBackend be = gui_backend();
    be.pump(waitMs);

    // Events, window by window (a snapshot: callbacks may open or close windows).
    struct Vec<&GuiWindow> snapshot = Vec::new();
    unsigned long i = 0UL;
    while (i < self.windows.len) { snapshot.push((*self.windows.at(i)).win); i = i + 1UL; }
    i = 0UL;
    while (i < snapshot.len) {
        &GuiWindow win = *snapshot.at(i);
        i = i + 1UL;
        ?&GuiAppWindow orec = __gui_app_rec(self, win);
        if (orec.is_null()) { continue; }
        &GuiAppWindow rec;
        unsafe { rec = (struct GuiAppWindow*)orec; }
        struct GuiEvent ev;
        while (rec.win.poll_event(&ev)) {
            int handled = 0;
            match (rec.ui) { case some(ui): { handled = ui.dispatch(&ev); } case null: { } }
            unsafe {
                if (!rec.onEvent.is_null()) {
                    fn int(&GuiApp, &GuiWindow, &GuiEvent, ?&Erased) cb = rec.onEvent!;
                    if (cb(self, rec.win, &ev, rec.onEventData)) { handled = 1; }
                }
            }
            int k = ev.kind;
            if (k == GUI_EVENT_CLOSE && !ev.defaultPrevented) { rec.closing = 1; }
            if (k == GUI_EVENT_RESIZE || k == GUI_EVENT_SCALE || k == GUI_EVENT_EXPOSE || handled) { rec.dirty = 1; }
            if (self.quitting) { break; }
            // a callback may have closed this very window
            if (__gui_app_rec(self, win).is_null()) { break; }
        }
    }
    snapshot.free();
    // Windows whose close button was clicked.
    i = 0UL;
    while (i < self.windows.len) {
        &GuiAppWindow r = *self.windows.at(i);
        if (r.closing) { self.close_window(r.win); continue; }
        i = i + 1UL;
    }

    // Timers that are due (a callback may add or cancel timers).
    now = self.now_ms();
    unsigned long n = self.timers.len;
    ti = 0UL;
    while (ti < n && ti < self.timers.len) {
        struct GuiTimer t = *self.timers.at(ti);
        if (!t.dead && t.due <= now + 0.5) {
            &GuiTimer tr = self.timers.at(ti);
            if (t.interval > 0.0) {
                tr.due = t.due + t.interval;
                if (tr.due < now) { tr.due = now + t.interval; }
            } else {
                tr.dead = 1;
            }
            unsafe {
                match (t.cb) { case some(cb): cb(self, t.id, t.data); case null: { } }
            }
            // a timer may have touched any window's state
            unsigned long w = 0UL;
            while (w < self.windows.len) { (*self.windows.at(w)).dirty = 1; w = w + 1UL; }
        }
        ti = ti + 1UL;
    }
    ti = 0UL;
    while (ti < self.timers.len) {
        if (self.timers.at(ti).dead) { self.timers.remove(ti); } else { ti = ti + 1UL; }
    }

    // Frames: animation callbacks and the UIs' clocks.
    now = self.now_ms();
    int frameDue = now - self.lastFrame >= self.frameMs - 0.5 ? 1 : 0;
    if (frameDue) { self.lastFrame = now; }
    i = 0UL;
    while (i < self.windows.len) {
        &GuiAppWindow r = *self.windows.at(i);
        i = i + 1UL;
        match (r.ui) { case some(ui): { ui.tick(now); } case null: { } }
        if (!frameDue) { continue; }
        unsafe {
            if (!r.onFrame.is_null()) {
                fn int(&GuiApp, &GuiWindow, double, ?&Erased) cb = r.onFrame!;
                if (!cb(self, r.win, now, r.onFrameData)) { r.onFrame = null; }
                r.dirty = 1;
            }
        }
    }

    // Repaint and present what changed.
    i = 0UL;
    while (i < self.windows.len) {
        &GuiAppWindow r = *self.windows.at(i);
        i = i + 1UL;
        int need = r.dirty;
        match (r.ui) { case some(ui): { if (ui.needs_redraw()) { need = 1; } } case null: { } }
        if (!need) { continue; }
        match (r.ui) { case some(ui): { ui.render(); } case null: { } }
        unsafe {
            if (!r.onDraw.is_null()) {
                if (!r.canvasReady) {
                    r.canvas = GuiCanvas::for_window(r.win);
                    r.canvasReady = 1;
                } else {
                    r.canvas.retarget_window(r.win);
                }
                r.canvas.reset();
                if (r.ui.is_null()) { r.canvas.clear(GuiColor::hex(0xFFFFFFU)); }
                fn void(&GuiApp, &GuiWindow, &GuiCanvas, ?&Erased) cb = r.onDraw!;
                cb(self, r.win, &r.canvas, r.onDrawData);
            }
        }
        r.win.present();
        r.dirty = 0;
    }
    return (self.quitting || self.windows.len == 0UL) ? 0 : 1;
}

int GuiApp::run() {
    while (self.pump(-1)) { }
    return self.exitCode;
}

void GuiApp::free() { self.drop(); }

void GuiApp::drop() {
    while (self.windows.len > 0UL) {
        &GuiAppWindow r = *self.windows.at(self.windows.len - 1UL);
        self.close_window(r.win);
    }
    self.windows.free();
    self.timers.free();
}

} // namespace std
