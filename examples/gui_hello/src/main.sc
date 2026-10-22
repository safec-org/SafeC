// gui_hello — a std/gui window with a few widgets and a vector-drawn
// banner, animated for about a second before it closes itself, so
// 'safeguard bundle' has a GUI program to package (and the bundle smoke
// test one that exits by itself).
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>

extern int printf(const char* fmt, ...);

struct Hello {
    &Widget progress;
    &Widget status;
    int frames;
    double start;
};

// The banner: a gradient card with the greeting, drawn on a canvas widget.
static void draw_banner(&Widget w, &GuiCanvas c, ?&Hello oh) {
    float fw = (float)w.w;
    float fh = (float)w.h;
    struct GuiPaint g = GuiPaint::linear(0.0f, 0.0f, fw, fh);
    g.add_stop(0.0f, GuiColor::hex(0x3B5BDBU));
    g.add_stop(1.0f, GuiColor::hex(0x9C36B5U));
    c.set_fill_paint(&g);
    c.fill_rounded_rect(0.0f, 0.0f, fw, fh, 10.0f);
    c.set_fill_color(GuiColor::hex(0xFFFFFFU));
    c.set_font_size(22.0f);
    c.set_text_align(GUI_TEXT_CENTER);
    c.set_text_baseline(GUI_BASELINE_MIDDLE);
    c.fill_text("Hello from SafeC", fw / 2.0f, fh / 2.0f);
}

// One animation frame: advance the progress bar; close after a second.
static int on_frame(&GuiApp app, &GuiWindow win, double now, ?&Hello oh) {
    match (oh) {
        case some(h): {
            if (h.frames == 0) { h.start = now; }
            h.frames = h.frames + 1;
            double t = (now - h.start) / 1000.0;
            h.progress.set_value(t);
            if (t >= 1.0) {
                app.close_window(win);
                return 0;
            }
        }
        case null: { }
    }
    return 1;
}

static void on_quit(&Widget w, ?&GuiApp oapp) {
    match (oapp) { case some(app): { app.quit(0); } case null: { } }
}

int main() {
    struct GuiApp app = GuiApp::new();
    ?&GuiWindow ow = app.open_window("Gui Hello", 360, 220);
    match (ow) {
        case null: {
            // No display to open a window on (a headless CI runner, say).
            unsafe { printf("gui_hello: no window (no display?)\n"); }
            return 0;
        }
        case some(win): {
            struct Hello h;
            &Widget root = Widget::vstack().set_padding(16);
            root.add_child(Widget::canvas(draw_banner, &h).set_pref_size(0, 90));
            h.progress = Widget::progress(0.0);
            root.add_child(h.progress);
            h.status = Widget::label("Closing by itself in a second...");
            root.add_child(h.status);
            root.add_child(Widget::button("Close now").on_click(on_quit, &app)
                                                     .set_alignment(WIDGET_ALIGN_END));
            h.frames = 0;
            h.start = 0.0;
            app.set_root(win, root);
            app.on_frame(win, on_frame, &h);
            app.run();
            app.free();
            unsafe { printf("gui_hello: window closed after %d frames\n", h.frames); }
        }
    }
    return 0;
}
