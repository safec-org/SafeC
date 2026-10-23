// gui_web — std/gui running in a web page (WebAssembly): two windows, each
// a <canvas> of web/index.html. "Controls" is a widget tree (buttons, a
// text field, a slider, a check box, a list); "Canvas" is the vector
// canvas, animated from GuiApp's frame callbacks. Same code as a desktop
// program, except that the page drives the loop: start() opens the
// windows and hands the app to gui_web_run() (see std/gui/gui_web.h).
//
//   safeguard build --target wasm32
//   cp build/gui_web web/gui_web.wasm
//   cp <SafeC>/std/gui/gui_web.js web/
//   python3 -m http.server -d web        # then open http://localhost:8000
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_web.h>
#include <std/collections/string.h>
#include <std/math.h>

struct Demo {
    &Widget status;
    &Widget label;
    &Widget progress;
    &Widget input;
    &Widget spin;
    int clicks;
    double t;
    ?&GuiWindow canvasWin;
};

static struct GuiApp gApp;
static struct Demo gDemo;

static void set_status2(&Demo d, const &char a, const &char b) {
    struct String s = String::from(a);
    s.push(b);
    d.status.set_text(s.data);
    s.free();
}

static void on_click(&Widget w, ?&Demo od) {
    match (od) {
        case some(d): {
            d.clicks = d.clicks + 1;
            struct String s = String::from("Clicked ");
            s.push_int((long long)d.clicks);
            if (d.clicks == 1) { s.push(" time"); } else { s.push(" times"); }
            d.label.set_text(s.data);
            s.free();
        }
        case null: { }
    }
}

static void on_slider(&Widget w, ?&Demo od) {
    match (od) { case some(d): { d.progress.set_value(w.value() / 100.0); } case null: { } }
}

static void on_input(&Widget w, ?&Demo od) {
    match (od) { case some(d): { set_status2(d, "Typed: ", w.text.data); } case null: { } }
}

static void on_spin(&Widget w, ?&Demo od) {
    match (od) {
        case some(d): {
            match (d.canvasWin) { case some(cw): { gApp.redraw(cw); } case null: { } }
        }
        case null: { }
    }
}

static void on_select(&Widget w, ?&Demo od) {
    match (od) {
        case some(d): {
            int i = w.selected_index();
            if (i >= 0) { unsafe { set_status2(d, "Selected: ", (const char*)w.item_text((unsigned long)i)); } }
        }
        case null: { }
    }
}

static &Widget build_controls(&Demo d) {
    &Widget root = Widget::vstack().set_padding(16).set_spacing(10);
    root.add_child(Widget::label("std/gui in a <canvas>"));
    &Widget row = Widget::hstack().set_spacing(10);
    row.add_child(Widget::button("Click me").on_click(on_click, d));
    d.label = Widget::label("Not clicked yet");
    row.add_child(d.label);
    root.add_child(row);
    d.input = Widget::textinput("Type here").on_change(on_input, d);
    root.add_child(d.input);
    &Widget sl = Widget::slider(0.0, 100.0, 40.0).on_change(on_slider, d);
    root.add_child(sl);
    d.progress = Widget::progress(0.4);
    root.add_child(d.progress);
    d.spin = Widget::checkbox("Spin the canvas", 1).on_change(on_spin, d);
    root.add_child(d.spin);
    &Widget list = Widget::list().on_change(on_select, d);
    list.add_item("WebAssembly");
    list.add_item("putImageData");
    list.add_item("requestAnimationFrame");
    list.add_item("devicePixelRatio");
    root.add_child(list);
    d.status = Widget::statusbar();
    d.status.set_text("Ready");
    root.add_child(d.status);
    return root;
}

static void draw_canvas(&GuiApp app, &GuiWindow win, &GuiCanvas c, ?&Demo od) {
    match (od) {
        case some(d): {
            float s = (float)win.scale;
            float w = (float)win.width / s;
            float h = (float)win.height / s;
            c.scale(s, s);
            c.set_fill_color(GuiColor::hex(0x14161CU));
            c.fill_rect(0.0f, 0.0f, w, h);
            float cx = w * 0.5f;
            float cy = h * 0.5f;
            int i = 0;
            while (i < 12) {
                double a = d.t * 0.0015 + (double)i * 3.14159265 / 6.0;
                float r = 40.0f + 30.0f * (float)sin_d(d.t * 0.002 + (double)i);
                c.set_fill_color(GuiColor::hsv((float)(i * 30), 0.7f, 0.95f, 0.85f));
                c.fill_circle(cx + (float)cos_d(a) * r * 1.6f, cy + (float)sin_d(a) * r * 1.6f, 14.0f);
                i = i + 1;
            }
            c.set_stroke_color(GuiColor::hex(0xE8E8F0U));
            c.set_line_width(3.0f);
            c.stroke_circle(cx, cy, 30.0f);
            c.set_fill_color(GuiColor::hex(0xFFFFFFU));
            c.set_font_size(18.0f);
            c.fill_text("SafeC on the web", 16.0f, 30.0f);
        }
        case null: { }
    }
}

static int on_frame(&GuiApp app, &GuiWindow win, double now, ?&Demo od) {
    match (od) {
        case some(d): { if (d.spin.is_checked()) { d.t = now; } }
        case null: { }
    }
    return 1;
}

// Called once by the page (gui_web.js): opens the windows and returns; the
// page then runs gApp from requestAnimationFrame.
__attribute__((export_name("start")))
int start() {
    gApp = GuiApp::new();
    gDemo.clicks = 0;
    gDemo.t = 0.0;
    ?&GuiWindow ow = gApp.open_window("Controls", 360, 420);
    gDemo.canvasWin = gApp.open_window("Canvas", 360, 420);
    match (ow) {
        case some(w): { gApp.set_root(w, build_controls(&gDemo)); }
        case null: { return 1; }
    }
    match (gDemo.canvasWin) {
        case some(cw): {
            gApp.on_draw(cw, draw_canvas, &gDemo);
            gApp.on_frame(cw, on_frame, &gDemo);
        }
        case null: { return 1; }
    }
    gui_web_run(&gApp);
    return 0;
}
