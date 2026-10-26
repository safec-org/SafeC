// The X11 GUI backend (std/gui/gui_x11.sc) against a real X server: draw,
// present and read the window back with XGetImage; send key, text, mouse,
// scroll, resize and close events through the server and poll them. Then
// two windows under one GuiApp: routing to the right window's widgets,
// titles, sizes, positions, painting, closing one, and the CLIPBOARD in
// both directions with another X client (a forked process).
#include <std/test/test.h>
#include <std/mem.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_gpu.h>
#include <std/collections/string.h>

extern void* XOpenDisplay(const char* name);
extern int XCloseDisplay(void* d);
extern int XSync(void* d, int discard);
extern void* XGetImage(void* d, unsigned long drawable, int x, int y, unsigned int w, unsigned int h,
                       unsigned long planes, int format);
extern int XDestroyImage(void* image);
extern int XSendEvent(void* d, unsigned long w, int propagate, long mask, void* event);
extern unsigned char XKeysymToKeycode(void* d, unsigned long keysym);
extern unsigned long XInternAtom(void* d, const char* name, int onlyIfExists);
extern unsigned long XDefaultRootWindow(void* d);
extern int XResizeWindow(void* d, unsigned long w, unsigned int width, unsigned int height);
extern int XGetWindowAttributes(void* d, unsigned long w, void* attrs);
extern int XGetWindowProperty(void* d, unsigned long w, unsigned long prop, long offset, long length,
                              int del, unsigned long reqType, unsigned long* actualType, int* actualFormat,
                              unsigned long* nitems, unsigned long* after, unsigned char** data);
extern int XFree(void* p);
extern unsigned long XCreateSimpleWindow(void* d, unsigned long parent, int x, int y, unsigned int w,
                                         unsigned int h, unsigned int bw, unsigned long border, unsigned long bg);
extern int XDestroyWindow(void* d, unsigned long w);
extern int XPending(void* d);
extern int XNextEvent(void* d, void* ev);
extern int XFlush(void* d);
extern int XCheckTypedWindowEvent(void* d, unsigned long w, int type, void* ev);
extern unsigned long XGetSelectionOwner(void* d, unsigned long selection);
extern int XSetSelectionOwner(void* d, unsigned long selection, unsigned long owner, unsigned long time);
extern int XConvertSelection(void* d, unsigned long selection, unsigned long target, unsigned long prop,
                             unsigned long requestor, unsigned long time);
extern int XChangeProperty(void* d, unsigned long w, unsigned long prop, unsigned long type, int format,
                           int mode, const void* data, int n);
extern int fork();
extern void _exit(int code);
extern int waitpid(int pid, int* status, int options);
extern int usleep(unsigned int us);
extern int strcmp(const char* a, const char* b);
extern int printf(const char* fmt, ...);
extern unsigned long strlen(const char* s);


static void* g_dpy = (void*)0;     // the test's own connection
static unsigned long g_win = 0UL;
static void* g_gui_dpy = (void*)0; // std/gui's own connection

// The Display* std/gui's X11 backend uses, from its native handles.
static void* gui_display_of(&GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return (void*)0;
        case some(st): {
            &GuiBackend b = gui_backend();
            struct GuiNativeHandle h = b.native_handle(st);
            unsafe { return (void*)(unsigned long)h.display; }
        }
    }
    return (void*)0;
}

// 0x00RRGGBB at (x, y) of the window, read back from the server.
static unsigned int read_pixel_of(unsigned long xw, int x, int y, int w, int h) {
    unsafe {
        void* img = XGetImage(g_dpy, xw, 0, 0, (unsigned int)w, (unsigned int)h, ~0UL, 2);
        if (img == (void*)0) { return 0xFFFFFFFFu; }
        unsigned char* data = *(unsigned char**)((unsigned long)img + 16UL);
        int bpl = *(int*)((unsigned long)img + 44UL);
        unsigned int px = *(unsigned int*)(data + (unsigned long)(y * bpl + x * 4));
        XDestroyImage(img);
        return px & 0xFFFFFFu;
    }
}

static unsigned int read_pixel(int x, int y, int w, int h) { return read_pixel_of(g_win, x, y, w, h); }

// Sends a 192-byte XEvent with the given head and int fields at 64..91.
static void send_input_to(unsigned long xw, int type, long mask, int x, int y, unsigned int state, unsigned int detail) {
    long ev[24];
    int i = 0;
    while (i < 24) { ev[i] = 0L; i = i + 1; }
    unsafe {
        unsigned char* b = (unsigned char*)&ev[0];
        *(int*)(b + 0) = type;
        *(unsigned long*)(b + 32) = xw;
        *(unsigned long*)(b + 40) = XDefaultRootWindow(g_dpy);
        *(int*)(b + 64) = x;
        *(int*)(b + 68) = y;
        *(unsigned int*)(b + 80) = state;
        *(unsigned int*)(b + 84) = detail;   // keycode / button
        *(int*)(b + 88) = 1;                 // same_screen
        XSendEvent(g_dpy, xw, 0, mask, (void*)b);
        XSync(g_dpy, 0);
    }
}

static void send_input(int type, long mask, int x, int y, unsigned int state, unsigned int detail) {
    send_input_to(g_win, type, mask, x, y, state, detail);
}

// Polls until an event of 'kind' arrives (others are skipped), or gives up.
static int wait_event(&GuiWindow win, int kind, &GuiEvent out) {
    int tries = 0;
    while (tries < 2000) {
        if (win.poll_event(out) && out.kind == kind) { return 1; }
        unsafe { XSync(g_dpy, 0); }
        tries = tries + 1;
    }
    return 0;
}

void t_x11() {
    unsafe {
        struct GuiWindow win = GuiWindow::create("x11 check", 64, 48);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        g_dpy = XOpenDisplay((const char*)0);
        ASSERT_TRUE(g_dpy != (void*)0);
        g_win = (unsigned long)win.native_handle();
        g_gui_dpy = gui_display_of(&win);
        XSync(g_gui_dpy, 0);

        win.fill_rect(0, 0, 64, 48, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
        win.fill_rect(10, 10, 20, 10, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        win.fill_rect(40, 30, 5, 5, GuiColor::rgb((unsigned char)0, (unsigned char)128, (unsigned char)255));
        win.present();
        XSync(g_gui_dpy, 0);
        test_assert_eq_u(read_pixel(15, 15, 64, 48), 0xFF0000ULL, "red rectangle on screen", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(42, 32, 64, 48), 0x0080FFULL, "R/G/B order", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(2, 2, 64, 48), 0x000000ULL, "background", __FILE__, __LINE__);
        int frame = 0;
        while (frame < 100) { win.present(); frame = frame + 1; }   // no per-frame allocation

        struct GuiEvent ev;
        send_input(2, 1L, 0, 0, 0u, (unsigned int)XKeysymToKeycode(g_dpy, 0xFF51UL));   // KeyPress Left
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_LEFT);
        ASSERT_EQ(ev.codepoint, 0);
        send_input(2, 1L, 0, 0, 0u, (unsigned int)XKeysymToKeycode(g_dpy, 0x61UL));     // 'a'
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.codepoint, 'a');
        send_input(2, 1L, 0, 0, 1u, (unsigned int)XKeysymToKeycode(g_dpy, 0x61UL));     // Shift+a
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.codepoint, 'A');
        send_input(3, 2L, 0, 0, 0u, (unsigned int)XKeysymToKeycode(g_dpy, 0xFF0DUL));   // KeyRelease Return
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_UP, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_ENTER);

        send_input(4, 4L, 5, 6, 0u, 1u);                       // ButtonPress, left
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_DOWN, &ev));
        ASSERT_TRUE(ev.x == 5.0 && ev.y == 6.0 && ev.button == 0);
        send_input(5, 8L, 7, 8, 0u, 3u);                       // ButtonRelease, right
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_UP, &ev));
        ASSERT_EQ(ev.button, 1);
        send_input(4, 4L, 1, 1, 0u, 5u);                       // wheel down
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_SCROLL, &ev));
        ASSERT_TRUE(ev.scrollDy == -1.0);
        send_input(6, 64L, 9, 10, 0u, 0u);                     // MotionNotify
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_MOVE, &ev));
        ASSERT_TRUE(ev.x == 9.0);

        XResizeWindow(g_dpy, g_win, 80u, 50u);
        XSync(g_dpy, 0);
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_RESIZE, &ev));
        ASSERT_EQ(ev.width, 80);
        ASSERT_EQ(win.width, 80);
        ASSERT_EQ(win.height, 50);
        win.fill_rect(70, 40, 5, 5, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present();
        XSync(g_gui_dpy, 0);
        test_assert_eq_u(read_pixel(72, 42, 80, 50), 0x00FF00ULL, "drawn after resize", __FILE__, __LINE__);

        long cm[24];
        int i = 0;
        while (i < 24) { cm[i] = 0L; i = i + 1; }
        unsigned char* b = (unsigned char*)&cm[0];
        *(int*)(b + 0) = 33;                                   // ClientMessage
        *(unsigned long*)(b + 32) = g_win;
        *(unsigned long*)(b + 40) = XInternAtom(g_dpy, "WM_PROTOCOLS", 0);
        *(int*)(b + 48) = 32;
        *(long*)(b + 56) = (long)XInternAtom(g_dpy, "WM_DELETE_WINDOW", 0);
        XSendEvent(g_dpy, g_win, 0, 0L, (void*)b);
        XSync(g_dpy, 0);
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_CLOSE, &ev));
        ASSERT_EQ(win.shouldClose, 1);

        win.destroy();
        ASSERT_TRUE(win.platform.is_null());
        XCloseDisplay(g_dpy);
    }
}


// ── several windows under one GuiApp ─────────────────────────────────────

static int g_clicks1 = 0;
static int g_clicks2 = 0;
static void click1(&Widget w, ?&void d) { g_clicks1 = g_clicks1 + 1; }
static void click2(&Widget w, ?&void d) { g_clicks2 = g_clicks2 + 1; }
static void paint2(&GuiApp app, &GuiWindow win, &GuiCanvas c, ?&void d) {
    c.set_fill_color(GuiColor::hex(0xFF0000U));
    c.fill_rect(0.0f, 0.0f, 8.0f, 8.0f);
}

// The server's idea of a window's size (XWindowAttributes: x, y, width, height).
static int server_width(unsigned long xw) {
    long attrs[32];
    unsafe {
        if (!XGetWindowAttributes(g_dpy, xw, (void*)&attrs[0])) { return -1; }
        return *(int*)((unsigned long)&attrs[0] + 8UL);
    }
}

// A UTF-8 window property as a String ("" if missing).
static int prop_is(unsigned long xw, const char* prop, const char* want) {
    unsafe {
        unsigned long type = 0UL; int format = 0; unsigned long n = 0UL; unsigned long after = 0UL;
        unsigned char* data = (unsigned char*)0;
        XGetWindowProperty(g_dpy, xw, XInternAtom(g_dpy, prop, 0), 0L, 4096L, 0, 0UL,
                           &type, &format, &n, &after, &data);
        if (data == (unsigned char*)0) { return 0; }
        int same = (n == strlen(want) && strcmp((const char*)data, want) == 0) ? 1 : 0;
        XFree((void*)data);
        return same;
    }
}

static void click_at(&GuiApp app, unsigned long xw, int x, int y, &int counter, int want) {
    send_input_to(xw, 4, 4L, x, y, 0u, 1u);        // ButtonPress, left
    send_input_to(xw, 5, 8L, x, y, 256u, 1u);      // ButtonRelease (Button1 held)
    int tries = 0;
    while (*counter < want && tries < 200) { app.pump(5); tries = tries + 1; }
}

// A forked X client that owns CLIPBOARD and serves one UTF8_STRING request.
static void clipboard_owner_child(const char* text) {
    unsafe {
        void* d = XOpenDisplay((const char*)0);
        if (d == (void*)0) { _exit(2); }
        unsigned long w = XCreateSimpleWindow(d, XDefaultRootWindow(d), 0, 0, 1u, 1u, 0u, 0UL, 0UL);
        unsigned long clip = XInternAtom(d, "CLIPBOARD", 0);
        unsigned long utf8 = XInternAtom(d, "UTF8_STRING", 0);
        XSetSelectionOwner(d, clip, w, 0UL);
        XSync(d, 0);
        int waited = 0;
        while (waited < 500) {
            while (XPending(d) > 0) {
                long ev[24];
                XNextEvent(d, (void*)&ev[0]);
                unsigned char* b = (unsigned char*)&ev[0];
                if (*(int*)b != 30) { continue; }          // SelectionRequest
                unsigned long requestor = *(unsigned long*)(b + 40);
                unsigned long target = *(unsigned long*)(b + 56);
                unsigned long property = *(unsigned long*)(b + 64);
                long reply[24];
                int i = 0;
                while (i < 24) { reply[i] = 0L; i = i + 1; }
                unsigned char* r = (unsigned char*)&reply[0];
                *(int*)r = 31;                              // SelectionNotify
                *(int*)(r + 16) = 1;
                *(unsigned long*)(r + 32) = requestor;
                *(unsigned long*)(r + 40) = clip;
                *(unsigned long*)(r + 48) = target;
                *(unsigned long*)(r + 56) = 0UL;
                if (target == utf8) {
                    XChangeProperty(d, requestor, property, utf8, 8, 0, (const void*)text, (int)strlen(text));
                    *(unsigned long*)(r + 56) = property;
                }
                XSendEvent(d, requestor, 0, 0L, (void*)r);
                XSync(d, 0);
                if (target == utf8) { _exit(0); }
            }
            usleep(10000u);
            waited = waited + 1;
        }
        _exit(3);
    }
}

void t_x11_multi() {
    unsafe {
        struct GuiApp app = GuiApp::new();
        struct GuiWindowOptions o1 = GuiWindowOptions::new("first", 140, 90);
        o1.x = 5; o1.y = 5;
        struct GuiWindowOptions o2 = GuiWindowOptions::new("second", 140, 90);
        o2.x = 160; o2.y = 5;
        ?&GuiWindow ow1 = app.open(&o1);
        ?&GuiWindow ow2 = app.open(&o2);
        ASSERT_FALSE(ow1.is_null());
        ASSERT_FALSE(ow2.is_null());
        if (ow1.is_null() || ow2.is_null()) { return; }
        &GuiWindow w1 = (struct GuiWindow*)ow1;
        &GuiWindow w2 = (struct GuiWindow*)ow2;
        g_dpy = XOpenDisplay((const char*)0);
        unsigned long x1 = (unsigned long)w1.native_handle();
        unsigned long x2 = (unsigned long)w2.native_handle();
        ASSERT_TRUE(x1 != 0UL && x2 != 0UL && x1 != x2);
        ASSERT_EQ(app.window_count(), 2UL);

        &Widget r1 = Widget::vstack().set_padding(10);
        &Widget b1 = Widget::button("One").on_click(click1, (?&void)null);
        r1.add_child(b1);
        &Widget r2 = Widget::vstack().set_padding(10);
        &Widget b2 = Widget::button("Two").on_click(click2, (?&void)null);
        r2.add_child(b2);
        app.set_root(w1, r1);
        app.set_root(w2, r2);
        app.on_draw(w2, paint2, (?&void)null);
        int i = 0;
        while (i < 10) { app.pump(5); i = i + 1; }
        XSync(g_gui_dpy, 0);
        // Both painted: window 2's draw callback on top of its widgets.
        test_assert_eq_u(read_pixel_of(x2, 2, 2, 140, 90), 0xFF0000ULL, "second window painted", __FILE__, __LINE__);
        ASSERT_TRUE(read_pixel_of(x1, 2, 2, 140, 90) != 0xFF0000U);

        // A click in window 2 reaches window 2's button only.
        click_at(&app, x2, b2.x + 5, b2.y + 5, &g_clicks2, 1);
        ASSERT_EQ(g_clicks2, 1);
        ASSERT_EQ(g_clicks1, 0);
        click_at(&app, x1, b1.x + 5, b1.y + 5, &g_clicks1, 1);
        ASSERT_EQ(g_clicks1, 1);
        ASSERT_EQ(g_clicks2, 1);
        // Keys too: Space on window 1's focused button.
        send_input_to(x1, 2, 1L, 0, 0, 0u, (unsigned int)XKeysymToKeycode(g_dpy, 0x20UL));
        i = 0;
        while (g_clicks1 < 2 && i < 200) { app.pump(5); i = i + 1; }
        ASSERT_EQ(g_clicks1, 2);
        ASSERT_EQ(g_clicks2, 1);

        // Titles (UTF-8), sizes, positions.
        w2.set_title("Second \xE2\x9C\x93");
        XSync(g_gui_dpy, 0);
        ASSERT_TRUE(prop_is(x2, "_NET_WM_NAME", "Second \xE2\x9C\x93"));
        ASSERT_TRUE(prop_is(x1, "_NET_WM_NAME", "first"));
        w1.set_size(180, 100);
        XSync(g_gui_dpy, 0);
        ASSERT_EQ(server_width(x1), 180);
        i = 0;
        while (w1.width != 180 && i < 200) { app.pump(5); i = i + 1; }
        ASSERT_EQ(w1.width, 180);
        ASSERT_EQ(w1.height, 100);
        ASSERT_EQ(server_width(x2), 140);
        w2.set_position(200, 30);
        ASSERT_EQ(w2.pos_x(), 200);

        // The clipboard, out: another client reads what we copied.
        ASSERT_TRUE(gui_clipboard_set_text("copied \xE2\x9C\x93"));
        unsigned long tw = XCreateSimpleWindow(g_dpy, XDefaultRootWindow(g_dpy), 0, 0, 1u, 1u, 0u, 0UL, 0UL);
        unsigned long clip = XInternAtom(g_dpy, "CLIPBOARD", 0);
        unsigned long utf8 = XInternAtom(g_dpy, "UTF8_STRING", 0);
        unsigned long prop = XInternAtom(g_dpy, "SAFEC_TEST_PASTE", 0);
        XConvertSelection(g_dpy, clip, utf8, prop, tw, 0UL);
        XFlush(g_dpy);
        long nev[24];
        int got = 0;
        i = 0;
        while (!got && i < 400) {
            app.pump(5);                                   // std/gui serves the request
            if (XCheckTypedWindowEvent(g_dpy, tw, 31, (void*)&nev[0])) { got = 1; }
            i = i + 1;
        }
        ASSERT_TRUE(got);
        ASSERT_TRUE(prop_is(tw, "SAFEC_TEST_PASTE", "copied \xE2\x9C\x93"));
        XDestroyWindow(g_dpy, tw);

        // The clipboard, in: paste what another client owns.
        unsigned long ours = XGetSelectionOwner(g_dpy, clip);
        int pid = fork();
        if (pid == 0) { clipboard_owner_child("from elsewhere \xC3\xA9"); }
        ASSERT_TRUE(pid > 0);
        i = 0;
        while (i < 300) {
            unsigned long owner = XGetSelectionOwner(g_dpy, clip);
            if (owner != 0UL && owner != ours) { break; }
            usleep(10000u);
            i = i + 1;
        }
        struct String pasted = String::new();
        ASSERT_TRUE(gui_clipboard_get_text(&pasted));
        ASSERT_TRUE(pasted.eq_cstr("from elsewhere \xC3\xA9"));
        pasted.free();
        int status = 0;
        waitpid(pid, &status, 0);
        ASSERT_EQ(status, 0);

        // Closing one window leaves the other running.
        long cm[24];
        int k = 0;
        while (k < 24) { cm[k] = 0L; k = k + 1; }
        unsigned char* b = (unsigned char*)&cm[0];
        *(int*)(b + 0) = 33;                               // ClientMessage
        *(unsigned long*)(b + 32) = x2;
        *(unsigned long*)(b + 40) = XInternAtom(g_dpy, "WM_PROTOCOLS", 0);
        *(int*)(b + 48) = 32;
        *(long*)(b + 56) = (long)XInternAtom(g_dpy, "WM_DELETE_WINDOW", 0);
        XSendEvent(g_dpy, x2, 0, 0L, (void*)b);
        XSync(g_dpy, 0);
        i = 0;
        while (app.window_count() > 1UL && i < 200) { app.pump(5); i = i + 1; }
        ASSERT_EQ(app.window_count(), 1UL);
        ASSERT_EQ(app.pump(0), 1);
        app.free();
        ASSERT_EQ(app.window_count(), 0UL);
        XCloseDisplay(g_dpy);
    }
}

// ── the OpenGL ES (EGL) presenter on the same window system ─────────────

void t_x11_gpu() {
    unsafe {
        struct GuiWindowOptions o = GuiWindowOptions::new("x11 gpu", 96, 64);
        o.present = GUI_PRESENT_GPU;
        o.hiDpi = 0;
        struct GuiWindow win = GuiWindow::create_with(&o);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        if (!gui_window_uses_gpu(&win)) {
            // No EGL/OpenGL ES driver on this machine: the CPU path took over.
            printf("  (no EGL/OpenGL ES here: GPU presentation skipped)\n");
            ASSERT_EQ(strcmp(gui_window_presenter_name(&win), "cpu"), 0);
            win.destroy();
            return;
        }
        ASSERT_EQ(strcmp(gui_window_presenter_name(&win), "gl"), 0);
        g_dpy = XOpenDisplay((const char*)0);
        g_win = (unsigned long)win.native_handle();
        win.fill_rect(0, 0, 96, 64, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
        win.fill_rect(10, 10, 20, 10, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        win.fill_rect(60, 40, 5, 5, GuiColor::rgb((unsigned char)0, (unsigned char)128, (unsigned char)255));
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(15, 15, 96, 64), 0xFF0000ULL, "gl: red rectangle on screen", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(62, 42, 96, 64), 0x0080FFULL, "gl: R/G/B order", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(2, 2, 96, 64), 0x000000ULL, "gl: background", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(95, 0, 96, 64), 0x000000ULL, "gl: not flipped (top right)", __FILE__, __LINE__);
        win.fill_rect(0, 60, 4, 4, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(1, 62, 96, 64), 0x00FF00ULL, "gl: bottom-left is the buffer's bottom-left", __FILE__, __LINE__);
        // partial updates: the damaged rectangle only
        win.fill_rect(40, 20, 8, 8, GuiColor::rgb((unsigned char)255, (unsigned char)255, (unsigned char)0));
        win.present_rect(40, 20, 8, 8);
        win.fill_rect(80, 5, 4, 4, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)255));
        win.present_rect(80, 5, 4, 4);
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(43, 23, 96, 64), 0xFFFF00ULL, "gl: partial update 1", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(81, 6, 96, 64), 0xFF00FFULL, "gl: partial update 2", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(15, 15, 96, 64), 0xFF0000ULL, "gl: the rest kept", __FILE__, __LINE__);
        int frame = 0;
        while (frame < 100) { win.present(); frame = frame + 1; }
        // a resize: a new texture of the new size
        struct GuiEvent ev;
        XResizeWindow(g_dpy, g_win, 120u, 70u);
        XSync(g_dpy, 0);
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_RESIZE, &ev));
        ASSERT_EQ(win.width, 120);
        win.fill_rect(0, 0, 120, 70, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)255));
        win.fill_rect(110, 60, 5, 5, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(112, 62, 120, 70), 0x00FF00ULL, "gl: drawn after resize", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(5, 5, 120, 70), 0x0000FFULL, "gl: whole new size", __FILE__, __LINE__);
        // back to the CPU path and to the GPU again, on the live window
        gui_window_use_cpu(&win);
        ASSERT_FALSE(gui_window_uses_gpu(&win));
        win.fill_rect(0, 0, 10, 10, GuiColor::rgb((unsigned char)255, (unsigned char)255, (unsigned char)255));
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(5, 5, 120, 70), 0xFFFFFFULL, "cpu again after gl", __FILE__, __LINE__);
        ASSERT_TRUE(gui_window_use_gpu(&win));
        win.fill_rect(0, 0, 10, 10, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel(5, 5, 120, 70), 0xFF0000ULL, "gl again", __FILE__, __LINE__);
        // a second GL window at the same time
        struct GuiWindow w2 = GuiWindow::create_with(&o);
        ASSERT_TRUE(gui_window_uses_gpu(&w2));
        w2.fill_rect(0, 0, 96, 64, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        w2.present();
        win.present();
        XSync(g_dpy, 0);
        test_assert_eq_u(read_pixel_of((unsigned long)w2.native_handle(), 50, 30, 96, 64), 0x00FF00ULL, "gl: second window", __FILE__, __LINE__);
        w2.destroy();
        win.destroy();
        XCloseDisplay(g_dpy);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("x11 backend: draw/present/events", t_x11);
        t.add("x11: two windows under GuiApp, titles, sizes, clipboard", t_x11_multi);
        t.add("x11: OpenGL ES presenter (EGL)", t_x11_gpu);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
