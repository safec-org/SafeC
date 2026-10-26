// std/gui on wasm32 (gui_web_wasm.sc), driven by run.mjs through the real
// std/gui/gui_web.js against a fake canvas DOM: three windows under one
// GuiApp — widgets, a custom-drawn one recording its events, and one that
// fits a page-sized canvas — plus exports run.mjs queries.
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_web.h>
#include <std/gui/gui_gpu.h>
#include <std/collections/string.h>

static struct GuiApp gApp;
static ?&GuiWindow gA = null;
static ?&GuiWindow gB = null;
static ?&GuiWindow gC = null;
static &Widget gButton;
static &Widget gInput;
static int gClicks = 0;
static int gTimer = 0;
static int gFrames = 0;
// window B's events
static int gKinds[20];
static double gLastX = 0.0;
static double gLastY = 0.0;
static double gLastDy = 0.0;
static int gLastKey = 0;
static int gLastCp = 0;
static int gLastMods = 0;
static int gLastButton = -1;

static void on_click(&Widget w, ?&Erased d) { gClicks = gClicks + 1; }

static int on_event_b(&GuiApp app, &GuiWindow win, &GuiEvent ev, ?&Erased d) {
    if (ev.kind >= 0 && ev.kind < 20) { gKinds[ev.kind] = gKinds[ev.kind] + 1; }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN || ev.kind == GUI_EVENT_MOUSE_MOVE || ev.kind == GUI_EVENT_SCROLL) {
        gLastX = ev.x;
        gLastY = ev.y;
    }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN) { gLastButton = ev.button; }
    if (ev.kind == GUI_EVENT_SCROLL) { gLastDy = ev.scrollDy; }
    if (ev.kind == GUI_EVENT_KEY_DOWN) {
        gLastKey = ev.keycode;
        gLastCp = (int)ev.codepoint;
        gLastMods = ev.modifiers;
    }
    return 0;
}

static void draw_b(&GuiApp app, &GuiWindow win, &GuiCanvas c, ?&Erased d) {
    c.set_fill_color(GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
    c.fill_rect(0.0f, 0.0f, (float)win.width, (float)win.height);
    // 50% white, premultiplied in the buffer: straight in the ImageData
    int y = 10;
    while (y < 14) {
        int x = 10;
        while (x < 14) { win.set_pixel(x, y, 0x80808080U); x = x + 1; }
        y = y + 1;
    }
}

static int frame_b(&GuiApp app, &GuiWindow win, double now, ?&Erased d) {
    gFrames = gFrames + 1;
    return 1;
}

static void tick(&GuiApp app, unsigned int id, ?&Erased d) { gTimer = gTimer + 1; }

__attribute__((export_name("start")))
int start() {
    gApp = GuiApp::new();
    gA = gApp.open_window("Widgets", 200, 150);
    gB = gApp.open_window("Events", 120, 80);
    struct GuiWindowOptions o = GuiWindowOptions::new("Fit", 50, 50);
    o.hiDpi = 0;
    gC = gApp.open(&o);
    if (gA.is_null() || gB.is_null() || gC.is_null()) { return 1; }
    unsafe {
        &GuiWindow a = (struct GuiWindow*)gA;
        &GuiWindow b = (struct GuiWindow*)gB;
        &Widget root = Widget::vstack().set_padding(10).set_spacing(8);
        gButton = Widget::button("Press").on_click(on_click, (?&Erased)null);
        gInput = Widget::textinput("type");
        root.add_child(gButton);
        root.add_child(gInput);
        gApp.set_root(a, root);
        gApp.on_draw(b, draw_b, (?&Erased)null);
        gApp.on_event(b, on_event_b, (?&Erased)null);
        gApp.on_frame(b, frame_b, (?&Erased)null);
        gApp.set_timeout(30.0, tick, (?&Erased)null);
    }
    gui_web_run(&gApp);
    return 0;
}

static ?&GuiWindow win_at(int i) {
    if (i == 0) { return gA; }
    if (i == 1) { return gB; }
    return gC;
}

__attribute__((export_name("win_id")))
int win_id(int i) { match (win_at(i)) { case some(w): return (int)w.id; case null: { } } return 0; }
__attribute__((export_name("win_w")))
int win_w(int i) { match (win_at(i)) { case some(w): return w.width; case null: { } } return 0; }
__attribute__((export_name("win_h")))
int win_h(int i) { match (win_at(i)) { case some(w): return w.height; case null: { } } return 0; }
__attribute__((export_name("win_scale1000")))
int win_scale1000(int i) { match (win_at(i)) { case some(w): return (int)(w.scale * 1000.0 + 0.5); case null: { } } return 0; }
__attribute__((export_name("win_lw")))
int win_lw(int i) { match (win_at(i)) { case some(w): return w.logical_width(); case null: { } } return 0; }
__attribute__((export_name("win_focused")))
int win_focused(int i) { match (win_at(i)) { case some(w): return w.is_focused(); case null: { } } return 0; }
__attribute__((export_name("set_size")))
void set_size_(int i, int w, int h) { match (win_at(i)) { case some(win): { win.set_size(w, h); } case null: { } } }
__attribute__((export_name("set_cursor")))
void set_cursor_(int i, int c) { match (win_at(i)) { case some(win): { win.set_cursor(c); } case null: { } } }
__attribute__((export_name("set_title")))
void set_title_(int i) { match (win_at(i)) { case some(win): { win.set_title("Renamed \xE2\x9C\x93"); } case null: { } } }

__attribute__((export_name("button_x")))
int button_x() { return gButton.x + gButton.w / 2; }
__attribute__((export_name("button_y")))
int button_y() { return gButton.y + gButton.h / 2; }
__attribute__((export_name("input_x")))
int input_x() { return gInput.x + 6; }
__attribute__((export_name("input_y")))
int input_y() { return gInput.y + gInput.h / 2; }
__attribute__((export_name("clicks")))
int clicks() { return gClicks; }
// 1 if the text input holds exactly the UTF-8 'want' (written into scratch by run.mjs).
static char gScratch[256];
__attribute__((export_name("scratch")))
char* scratch() { unsafe { return &gScratch[0]; } }
__attribute__((export_name("input_is")))
int input_is() { unsafe { return gInput.text.eq_cstr(&gScratch[0]) ? 1 : 0; } }
__attribute__((export_name("input_len")))
int input_len() { return (int)gInput.text.len; }

__attribute__((export_name("kind_count")))
int kind_count(int k) { return k >= 0 && k < 20 ? gKinds[k] : 0; }
__attribute__((export_name("last_x")))
double last_x() { return gLastX; }
__attribute__((export_name("last_y")))
double last_y() { return gLastY; }
__attribute__((export_name("last_dy")))
double last_dy() { return gLastDy; }
__attribute__((export_name("last_key")))
int last_key() { return gLastKey; }
__attribute__((export_name("last_cp")))
int last_cp() { return gLastCp; }
__attribute__((export_name("last_mods")))
int last_mods() { return gLastMods; }
__attribute__((export_name("last_button")))
int last_button() { return gLastButton; }
__attribute__((export_name("timer_fired")))
int timer_fired() { return gTimer; }
__attribute__((export_name("frames")))
int frames() { return gFrames; }
__attribute__((export_name("primary_mod")))
int primary_mod() { return gui_primary_mod(); }

// The clipboard from the program's side.
__attribute__((export_name("clip_set")))
int clip_set() { return gui_clipboard_set_text("copied \xE2\x9C\x93"); }
__attribute__((export_name("clip_is")))
int clip_is() {
    struct String s = String::new();
    int ok = gui_clipboard_get_text(&s);
    int same = 0;
    unsafe { same = ok && s.eq_cstr(&gScratch[0]) ? 1 : 0; }
    s.free();
    return same;
}

__attribute__((export_name("quit")))
void quit() { gApp.quit(0); }
__attribute__((export_name("close_b")))
void close_b() { match (gB) { case some(w): { w.request_close(); } case null: { } } }
__attribute__((export_name("window_count")))
int window_count() { return (int)gApp.window_count(); }

// windows in a page: positions (frames mode), dom.h interop, announcements
__attribute__((export_name("pos_x")))
int pos_x(int i) { match (win_at(i)) { case some(w): return w.pos_x(); case null: { } } return 0; }
__attribute__((export_name("pos_y")))
int pos_y(int i) { match (win_at(i)) { case some(w): return w.pos_y(); case null: { } } return 0; }
__attribute__((export_name("set_position")))
void set_position_(int i, int x, int y) { match (win_at(i)) { case some(w): { w.set_position(x, y); } case null: { } } }
__attribute__((export_name("elem")))
int elem(int i) { match (win_at(i)) { case some(w): return gui_web_window_element(w); case null: { } } return -1; }
__attribute__((export_name("mount")))
int mount(int i, int parent) { match (win_at(i)) { case some(w): return gui_web_mount(w, parent); case null: { } } return 0; }
__attribute__((export_name("announce")))
void announce() { gui_web_announce("Saved \xE2\x9C\x93"); }
__attribute__((export_name("uses_gpu")))
int uses_gpu(int i) { match (win_at(i)) { case some(w): return gui_window_uses_gpu(w); case null: { } } return 0; }
__attribute__((export_name("redraw_b")))
void redraw_b() { match (gB) { case some(w): { gApp.redraw(w); } case null: { } } }
// A partial present of window B: a 4x4 green square at (30, 20).
__attribute__((export_name("present_part")))
void present_part() {
    match (gB) {
        case some(w): {
            int y = 20;
            while (y < 24) {
                int x = 30;
                while (x < 34) { w.set_pixel(x, y, 0x00FF00FFU); x = x + 1; }
                y = y + 1;
            }
            w.present_rect(30, 20, 4, 4);
        }
        case null: { }
    }
}
