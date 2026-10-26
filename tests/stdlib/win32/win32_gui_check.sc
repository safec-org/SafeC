// The Win32 GUI backend (std/gui/gui_win32.sc): client size, pixels read
// back with GetPixel after present(), and posted key, text, mouse, wheel,
// resize and close messages coming out of poll_event(). Then two windows
// under one GuiApp: messages routed to the right window's widgets, titles,
// the clipboard, and closing one of them.
#include <std/test/test.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_gpu.h>
#include <std/collections/string.h>

struct Hwnd_ { char opaque_; };     // HWND
struct Hdc_ { char opaque_; };      // HDC
struct Hmodule_ { char opaque_; };  // HMODULE
struct ProcT_ { char opaque_; };    // a FARPROC, only compared with null
extern int PostMessageA(Hwnd_* hwnd, unsigned int msg, unsigned long long wparam, long long lparam);
extern Hdc_* GetDC(Hwnd_* hwnd);
extern int ReleaseDC(Hwnd_* hwnd, Hdc_* dc);
extern unsigned int GetPixel(Hdc_* dc, int x, int y);
extern int GetClientRect(Hwnd_* hwnd, &int rect);
extern int SetWindowPos(Hwnd_* hwnd, Hwnd_* after, int x, int y, int cx, int cy, unsigned int flags);
extern int GdiFlush();
extern int GetWindowTextA(Hwnd_* hwnd, char* buf, int n);
extern int strcmp(const char* a, const char* b);
extern int printf(const char* fmt, ...);
extern Hmodule_* GetModuleHandleA(const char* name);
extern ProcT_* GetProcAddress(Hmodule_* module, const char* name);

static Hwnd_* g_hwnd = (Hwnd_*)0;

// 0x00RRGGBB at client (x, y).
static unsigned int pixel_at(int x, int y) {
    unsafe {
        GdiFlush();
        Hdc_* dc = GetDC(g_hwnd);
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

static long long lp(int x, int y) { return (long long)(((y & 0xFFFF) << 16) | (x & 0xFFFF)); }

void t_win32_gui() {
    unsafe {
        struct GuiWindow win = GuiWindow::create("win32 check", 160, 100);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        g_hwnd = (Hwnd_*)win.native_handle();
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

        PostMessageA(g_hwnd, 0x0100u, 0x25ULL, 1LL);                       // WM_KEYDOWN VK_LEFT
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_LEFT);
        PostMessageA(g_hwnd, 0x0100u, 0x41ULL, 1LL);                       // WM_KEYDOWN 'A'
        PostMessageA(g_hwnd, 0x0102u, (unsigned long long)'a', 1LL);           // WM_CHAR 'a'
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_DOWN, &ev));
        ASSERT_EQ(ev.codepoint, 'a');                  // the typed text rides on the key press
        PostMessageA(g_hwnd, 0x0101u, 0x0DULL, 1LL);                       // WM_KEYUP Enter
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_KEY_UP, &ev));
        ASSERT_EQ(ev.keycode, GUI_KEY_ENTER);

        PostMessageA(g_hwnd, 0x0201u, 1ULL, lp(5, 6));                    // WM_LBUTTONDOWN
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_DOWN, &ev));
        ASSERT_TRUE(ev.x == 5.0 && ev.y == 6.0 && ev.button == 0);
        PostMessageA(g_hwnd, 0x0205u, 0ULL, lp(7, 8));                    // WM_RBUTTONUP
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_UP, &ev));
        ASSERT_EQ(ev.button, 1);
        PostMessageA(g_hwnd, 0x0200u, 0ULL, lp(-3, 9));                   // WM_MOUSEMOVE, left of the window
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_MOUSE_MOVE, &ev));
        ASSERT_TRUE(ev.x == -3.0 && ev.y == 9.0);
        PostMessageA(g_hwnd, 0x020Au, (unsigned long long)(0xFF88ULL << 16), lp(300, 300));  // WM_MOUSEWHEEL -120
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_SCROLL, &ev));
        ASSERT_TRUE(ev.scrollDy == -1.0);

        PostMessageA(g_hwnd, 0x0005u, 0ULL, lp(180, 120));                // WM_SIZE 180x120
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_RESIZE, &ev));
        ASSERT_EQ(ev.width, 180);
        ASSERT_EQ(win.width, 180);
        ASSERT_EQ(win.height, 120);
        win.fill_rect(0, 0, 180, 120, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present();                             // the buffer matches the new size

        PostMessageA(g_hwnd, 0x0010u, 0ULL, 0LL);                          // WM_CLOSE
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_CLOSE, &ev));
        ASSERT_EQ(win.shouldClose, 1);
        win.destroy();
        ASSERT_TRUE(win.platform.is_null());
    }
}

static int g_c1 = 0;
static int g_c2 = 0;
static void click1(&Widget w, ?&Erased d) { g_c1 = g_c1 + 1; }
static void click2(&Widget w, ?&Erased d) { g_c2 = g_c2 + 1; }

static void click_at(&GuiApp app, Hwnd_* hwnd, int x, int y, &int counter, int want) {
    unsafe {
        PostMessageA(hwnd, 0x0201u, 1ULL, lp(x, y));                   // WM_LBUTTONDOWN
        PostMessageA(hwnd, 0x0202u, 0ULL, lp(x, y));                   // WM_LBUTTONUP
    }
    int tries = 0;
    while (*counter < want && tries < 200) { app.pump(5); tries = tries + 1; }
}

void t_win32_multi() {
    unsafe {
        struct GuiApp app = GuiApp::new();
        ?&GuiWindow ow1 = app.open_window("first", 160, 100);
        ?&GuiWindow ow2 = app.open_window("second", 160, 100);
        ASSERT_FALSE(ow1.is_null() || ow2.is_null());
        if (ow1.is_null() || ow2.is_null()) return;
        &GuiWindow w1 = (struct GuiWindow*)ow1;
        &GuiWindow w2 = (struct GuiWindow*)ow2;
        Hwnd_* h1 = (Hwnd_*)w1.native_handle();
        Hwnd_* h2 = (Hwnd_*)w2.native_handle();
        ASSERT_TRUE(h1 != h2);
        &Widget r1 = Widget::vstack().set_padding(10);
        &Widget b1 = Widget::button("One").on_click(click1, (?&Erased)null);
        r1.add_child(b1);
        &Widget r2 = Widget::vstack().set_padding(10);
        &Widget b2 = Widget::button("Two").on_click(click2, (?&Erased)null);
        r2.add_child(b2);
        app.set_root(w1, r1);
        app.set_root(w2, r2);
        int i = 0;
        while (i < 5) { app.pump(5); i = i + 1; }
        ASSERT_TRUE(b2.w > 0);
        click_at(&app, h2, b2.x + 5, b2.y + 5, &g_c2, 1);
        ASSERT_EQ(g_c2, 1);
        ASSERT_EQ(g_c1, 0);
        click_at(&app, h1, b1.x + 5, b1.y + 5, &g_c1, 1);
        ASSERT_EQ(g_c1, 1);
        ASSERT_EQ(g_c2, 1);
        w2.set_title("renamed");
        char buf[64];
        GetWindowTextA(h2, &buf[0], 64);
        ASSERT_EQ(strcmp((const char*)&buf[0], "renamed"), 0);
        ASSERT_TRUE(gui_clipboard_set_text("clip \xC3\xA9"));
        struct String s = String::new();
        ASSERT_TRUE(gui_clipboard_get_text(&s));
        ASSERT_TRUE(s.eq_cstr("clip \xC3\xA9"));
        s.free();
        PostMessageA(h2, 0x0010u, 0ULL, 0LL);                          // WM_CLOSE
        i = 0;
        while (app.window_count() > 1UL && i < 200) { app.pump(5); i = i + 1; }
        ASSERT_EQ(app.window_count(), 1UL);
        app.free();
    }
}

// The Direct3D 11 presenter (flip-model swap chain), where the system has
// one (Wine: wined3d over OpenGL; Windows: any D3D11 driver or WARP).
void t_win32_gpu() {
    unsafe {
        struct GuiWindowOptions o = GuiWindowOptions::new("win32 gpu", 160, 100);
        o.present = GUI_PRESENT_GPU;
        o.hiDpi = 0;
        struct GuiWindow win = GuiWindow::create_with(&o);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        g_hwnd = (Hwnd_*)win.native_handle();
        struct GuiEvent ev;
        while (win.poll_event(&ev)) { }
        if (!gui_window_uses_gpu(&win)) {
            printf("  (no Direct3D 11 here: GPU presentation skipped)\n");
            win.destroy();
            return;
        }
        ASSERT_EQ(strcmp(gui_window_presenter_name(&win), "d3d11"), 0);
        win.fill_rect(0, 0, 160, 100, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)0));
        win.fill_rect(10, 10, 30, 20, GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0));
        win.fill_rect(100, 60, 10, 10, GuiColor::rgb((unsigned char)0, (unsigned char)128, (unsigned char)255));
        win.present();
        int frame = 0;
        while (frame < 30) { win.present(); win.poll_event(&ev); frame = frame + 1; }
        ASSERT_TRUE(gui_window_uses_gpu(&win));   // no device loss, no fallback
        unsigned int red = pixel_at(20, 20);
        unsigned int blue = pixel_at(105, 65);
        // Wine composites the swap chain into the window GDI reads from;
        // Windows' DWM keeps flip-model frames out of GetPixel's reach.
        int wine = GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version") != (ProcT_*)0 ? 1 : 0;
        if (wine) {
            ASSERT_EQ(red, 0xFF0000U);
            ASSERT_EQ(blue, 0x0080FFU);
        }
        win.fill_rect(50, 40, 8, 8, GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0));
        win.present_rect(50, 40, 8, 8);
        frame = 0;    // (Wine's presents complete asynchronously, a frame or two later)
        while (frame < 10) { win.present_rect(50, 40, 8, 8); win.poll_event(&ev); frame = frame + 1; }
        if (wine) {
            ASSERT_EQ(pixel_at(53, 43), 0x00FF00U);   // the partial update
            ASSERT_EQ(pixel_at(20, 20), 0xFF0000U);   // and the rest of the frame
        }
        SetWindowPos(g_hwnd, (Hwnd_*)0, 0, 0, 260, 180, 0x0002U | 0x0004U);   // NOMOVE | NOZORDER
        ASSERT_TRUE(wait_event(&win, GUI_EVENT_RESIZE, &ev));
        win.fill_rect(0, 0, win.width, win.height, GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)255));
        win.present();
        win.present();
        ASSERT_TRUE(gui_window_uses_gpu(&win));   // ResizeBuffers worked
        gui_window_use_cpu(&win);
        win.present();
        ASSERT_EQ(pixel_at(5, 5), 0x0000FFU);     // GDI again after the swap chain
        win.destroy();
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("win32 gui backend: present/events", t_win32_gui);
        t.add("win32: two windows under GuiApp, titles, clipboard", t_win32_multi);
        t.add("win32: Direct3D 11 presenter", t_win32_gpu);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
