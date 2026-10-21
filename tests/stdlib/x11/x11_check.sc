// The X11 GUI backend (std/gui/gui_x11.sc) against a real X server: draw,
// present and read the window back with XGetImage; send key, text, mouse,
// scroll, resize and close events through the server and poll them.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>

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

// The head of gui_x11.sc's platform struct.
struct X11PlatformHead {
    void* display;
    unsigned long window;
};

static void* g_dpy = (void*)0;     // the test's own connection
static unsigned long g_win = 0UL;

// 0x00RRGGBB at (x, y) of the window, read back from the server.
static unsigned int read_pixel(int x, int y, int w, int h) {
    unsafe {
        void* img = XGetImage(g_dpy, g_win, 0, 0, (unsigned int)w, (unsigned int)h, ~0UL, 2);
        if (img == (void*)0) { return 0xFFFFFFFFu; }
        unsigned char* data = *(unsigned char**)((unsigned long)img + 16UL);
        int bpl = *(int*)((unsigned long)img + 44UL);
        unsigned int px = *(unsigned int*)(data + (unsigned long)(y * bpl + x * 4));
        XDestroyImage(img);
        return px & 0xFFFFFFu;
    }
}

// Sends a 192-byte XEvent with the given head and int fields at 64..91.
static void send_input(int type, long mask, int x, int y, unsigned int state, unsigned int detail) {
    long ev[24];
    int i = 0;
    while (i < 24) { ev[i] = 0L; i = i + 1; }
    unsafe {
        unsigned char* b = (unsigned char*)&ev[0];
        *(int*)(b + 0) = type;
        *(unsigned long*)(b + 32) = g_win;
        *(unsigned long*)(b + 40) = XDefaultRootWindow(g_dpy);
        *(int*)(b + 64) = x;
        *(int*)(b + 68) = y;
        *(unsigned int*)(b + 80) = state;
        *(unsigned int*)(b + 84) = detail;   // keycode / button
        *(int*)(b + 88) = 1;                 // same_screen
        XSendEvent(g_dpy, g_win, 0, mask, (void*)b);
        XSync(g_dpy, 0);
    }
}

// Polls until an event of 'kind' arrives (others are skipped), or gives up.
static int wait_event(&GuiWindow win, int kind, &GuiEvent out) {
    int tries = 0;
    while (tries < 2000) {
        if (gui_poll_event(win, out) && out.kind == kind) { return 1; }
        unsafe { XSync(g_dpy, 0); }
        tries = tries + 1;
    }
    return 0;
}

void t_x11() {
    unsafe {
        struct GuiWindow win = gui_create_window("x11 check", 64, 48);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        g_dpy = XOpenDisplay((const char*)0);
        ASSERT_TRUE(g_dpy != (void*)0);
        g_win = ((struct X11PlatformHead*)win.platform)->window;
        XSync(((struct X11PlatformHead*)win.platform)->display, 0);

        gui_fill_rect(&win, 0, 0, 64, 48, gui_rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
        gui_fill_rect(&win, 10, 10, 20, 10, gui_rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        gui_fill_rect(&win, 40, 30, 5, 5, gui_rgb((unsigned char)0, (unsigned char)128, (unsigned char)255));
        gui_present(&win);
        XSync(((struct X11PlatformHead*)win.platform)->display, 0);
        test_assert_eq_u(read_pixel(15, 15, 64, 48), 0xFF0000ULL, "red rectangle on screen", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(42, 32, 64, 48), 0x0080FFULL, "R/G/B order", __FILE__, __LINE__);
        test_assert_eq_u(read_pixel(2, 2, 64, 48), 0x000000ULL, "background", __FILE__, __LINE__);
        int frame = 0;
        while (frame < 100) { gui_present(&win); frame = frame + 1; }   // no per-frame allocation

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
        gui_fill_rect(&win, 70, 40, 5, 5, gui_rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        gui_present(&win);
        XSync(((struct X11PlatformHead*)win.platform)->display, 0);
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

        gui_destroy_window(&win);
        ASSERT_TRUE(win.platform.is_null());
        XCloseDisplay(g_dpy);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe { t.add("x11 backend: draw/present/events", t_x11); }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
