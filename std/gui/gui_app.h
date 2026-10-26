#pragma once
// SafeC Standard Library — GUI: the application object.
//
// A GuiApp owns any number of windows and runs them from one event loop:
// it pumps the platform's events, routes each to the window it belongs to
// (to that window's widget tree and/or your event callback), fires timers
// and per-frame animation callbacks, repaints the windows that changed and
// presents them, and closes a window when its close button is clicked
// (unless the GUI_EVENT_CLOSE is prevent_default()ed).
//
//   struct GuiApp app = GuiApp::new();
//   ?&GuiWindow ow = app.open_window("Hello", 480, 320);
//   match (ow) {
//       case some(win): { app.set_root(win, build_ui()); }
//       case null: { return 1; }       // no display
//   }
//   int code = app.run();             // until the last window closes or quit()
//   app.free();
//
// Driving the loop yourself (a game loop, an embedding host, or a
// WebAssembly page that can't block): call app.pump(0) once per frame —
// it does one iteration without waiting and returns 0 once the app should
// stop. app.pump(timeoutMs) with a positive timeout waits up to that long
// for the next event, timer or frame.
//
// Rendering: a window with a widget root is laid out and painted by its
// GuiUi; a window with an on_draw callback gets a GuiCanvas over its
// buffer (after the widgets, if it has both); a window with neither shows
// whatever you put in win.pixels yourself. Windows are repainted only
// when something changed: an input event, a resize, a timer or animation
// callback, or app.redraw(win). on_frame callbacks run once per frame
// (frame_interval(), 60 Hz by default) while they return 1, with the
// frame time in milliseconds — the animation API.
//
// Timers: set_timeout(ms, cb, data) runs cb once, set_interval(ms, ...)
// every ms until cancel_timer(id) (ids are never 0). They run on the loop
// (never concurrently with event handling), as close to their due time as
// the loop allows.

#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/collections/vec.h>

namespace std {

// One window and what the app does with it (internal; see the methods).
struct GuiAppWindow {
    &GuiWindow win;
    ?&GuiUi ui;
    void* onEvent;  void* onEventData;    // fn int(&GuiApp, &GuiWindow, &GuiEvent, ?&T)
    void* onDraw;   void* onDrawData;     // fn void(&GuiApp, &GuiWindow, &GuiCanvas, ?&T)
    void* onFrame;  void* onFrameData;    // fn int(&GuiApp, &GuiWindow, double, ?&T)
    struct GuiCanvas canvas;
    int canvasReady;
    int dirty;
    int closing;
};

struct GuiTimer {
    unsigned int id;
    double due;
    double interval;      // 0: one-shot
    void* cb;             // fn void(&GuiApp, unsigned int, ?&T)
    void* data;
    int dead;
};

struct GuiApp {
    impl !Send + !Sync;   // lives on the thread that created it (the UI thread)
    struct Vec<&GuiAppWindow> windows;
    struct Vec<struct GuiTimer> timers;
    unsigned int nextTimer;
    int quitting;
    int exitCode;
    double frameMs;
    double lastFrame;

    static struct GuiApp new();

    // ── windows ──────────────────────────────────────────────────────────
    // Opens a window (hiDpi). null if there's no display / it failed.
    ?&GuiWindow open_window(const &char title, int width, int height);
    ?&GuiWindow open(const &GuiWindowOptions opts);
    // Destroys the window (and its widget tree, if any) now.
    void close_window(&GuiWindow win);
    unsigned long window_count() const;
    ?&GuiWindow window_at(unsigned long i) const;
    ?&GuiWindow find_window(unsigned int id) const;
    // Shows 'root' in the window (the window owns the tree from now on:
    // closing the window frees it). Returns the window's GuiUi.
    &GuiUi set_root(&GuiWindow win, &Widget root);
    ?&GuiUi ui_of(const &GuiWindow win) const;
    // Repaint the window on the next iteration.
    void redraw(&GuiWindow win);

    // ── callbacks ────────────────────────────────────────────────────────
    // Every event for the window (after its widgets saw it): return 1 if
    // you handled it (the window is repainted).
    generic<T> void on_event(&GuiWindow win, fn int(&GuiApp, &GuiWindow, &GuiEvent, ?&T) cb, ?&T data);
    // Custom drawing, each repaint.
    generic<T> void on_draw(&GuiWindow win, fn void(&GuiApp, &GuiWindow, &GuiCanvas, ?&T) cb, ?&T data);
    // Animation: called every frame with the time in ms while it returns 1.
    generic<T> void on_frame(&GuiWindow win, fn int(&GuiApp, &GuiWindow, double, ?&T) cb, ?&T data);
    void on_event_raw_(&GuiWindow win, void* cb, void* data);
    void on_draw_raw_(&GuiWindow win, void* cb, void* data);
    void on_frame_raw_(&GuiWindow win, void* cb, void* data);

    // ── timers ───────────────────────────────────────────────────────────
    generic<T> unsigned int set_timeout(double ms, fn void(&GuiApp, unsigned int, ?&T) cb, ?&T data);
    generic<T> unsigned int set_interval(double ms, fn void(&GuiApp, unsigned int, ?&T) cb, ?&T data);
    unsigned int add_timer_raw_(double ms, int repeat, void* cb, void* data);
    void cancel_timer(unsigned int id);

    // ── the loop ─────────────────────────────────────────────────────────
    // One iteration: wait up to 'timeoutMs' (0 = don't wait, negative = as
    // long as nothing is due) for events, route them, run due timers and
    // frame callbacks, repaint and present what changed. Returns 0 once the
    // app should stop (quit() was called or no window is left).
    int  pump(int timeoutMs);
    // pump() until it returns 0; returns the exit code given to quit()
    // (0 when the last window closed).
    int  run();
    void quit(int exitCode);
    // gui_time_ms()
    double now_ms() const;
    void set_frame_interval(double ms);
    double frame_interval() const;
    // Closes every window and frees the app's state.
    void drop();
    void free() consume;   // = drop()
};

generic<T>
void GuiApp::on_event(&GuiWindow win, fn int(&GuiApp, &GuiWindow, &GuiEvent, ?&T) cb, ?&T data) {
    unsafe { self.on_event_raw_(win, (void*)cb, (void*)data); }
}

generic<T>
void GuiApp::on_draw(&GuiWindow win, fn void(&GuiApp, &GuiWindow, &GuiCanvas, ?&T) cb, ?&T data) {
    unsafe { self.on_draw_raw_(win, (void*)cb, (void*)data); }
}

generic<T>
void GuiApp::on_frame(&GuiWindow win, fn int(&GuiApp, &GuiWindow, double, ?&T) cb, ?&T data) {
    unsafe { self.on_frame_raw_(win, (void*)cb, (void*)data); }
}

generic<T>
unsigned int GuiApp::set_timeout(double ms, fn void(&GuiApp, unsigned int, ?&T) cb, ?&T data) {
    unsafe { return self.add_timer_raw_(ms, 0, (void*)cb, (void*)data); }
}

generic<T>
unsigned int GuiApp::set_interval(double ms, fn void(&GuiApp, unsigned int, ?&T) cb, ?&T data) {
    unsafe { return self.add_timer_raw_(ms, 1, (void*)cb, (void*)data); }
}

} // namespace std
