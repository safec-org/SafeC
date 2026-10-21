// The Win32 GUI backend (std/gui/gui_win32.sc): client size, pixels read
// back with GetPixel after present(), and posted key, text, mouse, wheel,
// resize and close messages coming out of poll_event().
#include <std/test/test.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>

extern int PostMessageA(void* hwnd, unsigned int msg, unsigned long wparam, long lparam);
extern void* GetDC(void* hwnd);
extern int ReleaseDC(void* hwnd, void* dc);
extern unsigned int GetPixel(void* dc, int x, int y);
extern int GetClientRect(void* hwnd, &int rect);
extern int SetWindowPos(void* hwnd, void* after, int x, int y, int cx, int cy, unsigned int flags);
extern int GdiFlush();

struct Win32PlatformHead { void* hwnd; };

static void* g_hwnd = (void*)0;

// 0x00RRGGBB at client (x, y).
static unsigned int pixel_at(int x, int y) {
    unsafe {
        GdiFlush();
        void* dc = GetDC(g_hwnd);
        unsigned int c = GetPixel(dc, x, y);      // COLORREF 0x00BBGGRR
        ReleaseDC(g_hwnd, dc);
        return ((c & 0xFFu) << 16) | (c & 0xFF00u) | ((c >> 16) & 0xFFu);
    }
}

static int wait_event(&GuiWindow win, int kind, &GuiEvent out) {
    int tries = 0;
    while (tries < 200) {
        if (win.poll_event(out) && out.kind == kind) { return 1; }
        tries = tries + 1;
    }
    return 0;
}

static long lp(int x, int y) { return (long)(((y & 0xFFFF) << 16) | (x & 0xFFFF)); }

void t_win32_gui() {
    unsafe {
        struct GuiWindow win = GuiWindow::create("win32 check", 160, 100);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        g_hwnd = ((struct Win32PlatformHead*)win.platform)->hwnd;
        int rc[4];
        GetClientRect(g_hwnd, &rc[0]);
        ASSERT_EQ(rc[2] - rc[0], 160);                 // the client area is the requested size
        // (Windows enforces a minimum window width; below it, the WM_SIZE
        // at creation resizes the buffer to the real client area.)
        ASSERT_EQ(rc[3] - rc[1], 100);
        struct GuiEvent ev;
        while (win.poll_event(&ev)) { }          // creation-time messages

        win.fill_rect(0, 0, 160, 100, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
        win.fill_rect(10, 10, 20, 10, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        win.fill_rect(40, 30, 5, 5, GuiColor::rgb((unsigned char)0, (unsigned char)128, (unsigned char)255));
        win.present();
        test_assert_eq_u(pixel_at(15, 15), 0xFF0000ULL, "red rectangle on screen", __FILE__, __LINE__);
        test_assert_eq_u(pixel_at(42, 32), 0x0080FFULL, "R/G/B order", __FILE__, __LINE__);
        test_assert_eq_u(pixel_at(2, 2), 0x000000ULL, "background", __FILE__, __LINE__);

        PostMessageA(g_hwnd, 0x0100u, 0x25UL, 1L);                       // WM_KEYDOWN VK_LEFT
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_LEFT);
        PostMessageA(g_hwnd, 0x0100u, 0x41UL, 1L);                       // WM_KEYDOWN 'A'
        PostMessageA(g_hwnd, 0x0102u, (unsigned long)'a', 1L);           // WM_CHAR 'a'
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.codepoint, 'a');                  // the typed text rides on the key press
        PostMessageA(g_hwnd, 0x0101u, 0x0DUL, 1L);                       // WM_KEYUP Enter
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_UP, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_ENTER);

        PostMessageA(g_hwnd, 0x0201u, 1UL, lp(5, 6));                    // WM_LBUTTONDOWN
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_DOWN, &ev));
        ASSERT_TRUE(ev.x == 5.0 && ev.y == 6.0 && ev.button == 0);
        PostMessageA(g_hwnd, 0x0205u, 0UL, lp(7, 8));                    // WM_RBUTTONUP
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_UP, &ev));
        ASSERT_EQ(ev.button, 1);
        PostMessageA(g_hwnd, 0x0200u, 0UL, lp(-3, 9));                   // WM_MOUSEMOVE, left of the window
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_MOVE, &ev));
        ASSERT_TRUE(ev.x == -3.0 && ev.y == 9.0);
        PostMessageA(g_hwnd, 0x020Au, (unsigned long)(0xFF88UL << 16), lp(300, 300));  // WM_MOUSEWHEEL -120
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_SCROLL, &ev));
        ASSERT_TRUE(ev.scrollDy == -1.0);

        PostMessageA(g_hwnd, 0x0005u, 0UL, lp(180, 120));                // WM_SIZE 180x120
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_RESIZE, &ev));
        ASSERT_EQ(ev.width, 180);
        ASSERT_EQ(win.width, 180);
        ASSERT_EQ(win.height, 120);
        win.fill_rect(0, 0, 180, 120, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present();                             // the buffer matches the new size

        PostMessageA(g_hwnd, 0x0010u, 0UL, 0L);                          // WM_CLOSE
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_CLOSE, &ev));
        ASSERT_EQ(win.shouldClose, 1);
        win.destroy();
        ASSERT_TRUE(win.platform.is_null());
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe { t.add("win32 gui backend: present/events", t_win32_gui); }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
