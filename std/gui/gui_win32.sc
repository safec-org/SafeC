// SafeC Standard Library — GUI backend: Windows (Win32 API). See gui.h for
// the portable API this implements.
//
// NOTE ON VERIFICATION: unlike gui_cocoa.sc (built and run for real on a
// live Mac while developing this library), this backend has been written
// carefully against the real Win32 API surface (RegisterClassExW,
// CreateWindowExW, GetMessage/PeekMessage's WNDPROC-callback event model,
// StretchDIBits for presenting a top-down 32bpp DIB) but could not be
// compiled or executed in the environment this was developed in (no
// Windows host/toolchain available). Treat it as a solid, API-correct
// starting point that hasn't had the same "actually ran it" pass the
// Cocoa backend got — sanity-check it on a real Windows machine before
// depending on it.
//
// Event model translation: Win32 delivers input via window-procedure
// callbacks (WndProc), not a poll-a-queue-yourself model like Cocoa/X11 —
// so WndProc here just pushes a GuiEvent into a small fixed-size ring
// buffer per window, and gui_poll_event() pops from that ring (pumping
// PeekMessage/DispatchMessage first so WndProc actually gets called).
#pragma once
#include <std/gui/gui.h>
#include <std/mem.sc>

namespace std {

// 'A' (ANSI, plain 'const char*') API variants used throughout, not the
// 'W' (wide-string/UTF-16) ones — avoids needing MultiByteToWideChar just
// to register a class name or set a window title, at the cost of non-
// Unicode title text, an acceptable trade for this backend's scope.
extern void* GetModuleHandleA(const void* moduleName);
extern unsigned short RegisterClassExA(const void* wndClassEx);
extern void* CreateWindowExA(unsigned int exStyle, const void* className, const void* windowName,
                              unsigned int style, int x, int y, int w, int h,
                              void* parent, void* menu, void* instance, void* param);
extern int ShowWindow(void* hwnd, int cmdShow);
extern int UpdateWindow(void* hwnd);
extern int PeekMessageA(void* msg, void* hwnd, unsigned int msgMin, unsigned int msgMax, unsigned int remove);
extern int TranslateMessage(const void* msg);
extern void* DispatchMessageA(const void* msg);
extern void* DefWindowProcA(void* hwnd, unsigned int msg, unsigned long wparam, long lparam);
extern void* LoadCursorA(void* instance, const void* cursorName);
extern void* GetDC(void* hwnd);
extern int ReleaseDC(void* hwnd, void* dc);
extern int StretchDIBits(void* dc, int destX, int destY, int destW, int destH,
                          int srcX, int srcY, int srcW, int srcH,
                          const void* bits, const void* bitmapInfo,
                          unsigned int usage, unsigned int rop);
extern int DestroyWindow(void* hwnd);
extern void PostQuitMessage(int exitCode);
extern void* SetWindowLongPtrW(void* hwnd, int index, void* newLong);
extern void* GetWindowLongPtrW(void* hwnd, int index);

#define GUI_WM_CLOSE       0x0010U
#define GUI_WM_DESTROY     0x0002U
#define GUI_WM_SIZE        0x0005U
#define GUI_WM_LBUTTONDOWN 0x0201U
#define GUI_WM_LBUTTONUP   0x0202U
#define GUI_WM_RBUTTONDOWN 0x0204U
#define GUI_WM_RBUTTONUP   0x0205U
#define GUI_WM_MOUSEMOVE   0x0200U
#define GUI_WM_KEYDOWN     0x0100U
#define GUI_WM_KEYUP       0x0101U
#define GUI_WM_MOUSEWHEEL  0x020AU
#define GUI_GWLP_USERDATA  (-21)

#define GUI_WIN32_QUEUE_CAP 64

// Mirrors WNDCLASSEXA/MSG's real field layout directly (rather than a
// hand-packed byte/word array) so standard C struct alignment rules give
// an ABI-compatible layout without manual offset arithmetic.
struct Win32WndClassExA {
    unsigned int cbSize;
    unsigned int style;
    void* lpfnWndProc;
    int cbClsExtra;
    int cbWndExtra;
    void* hInstance;
    void* hIcon;
    void* hCursor;
    void* hbrBackground;
    const char* lpszMenuName;
    const char* lpszClassName;
    void* hIconSm;
};

struct Win32Msg {
    void* hwnd;
    unsigned int message;
    unsigned long wParam;
    long lParam;
    unsigned int time;
    int ptX;
    int ptY;
};

struct GuiWin32Platform {
    void* hwnd;
    struct GuiEvent queue[GUI_WIN32_QUEUE_CAP];
    int queueHead;
    int queueTail;
    int closeRequested;
    struct GuiWindow* selfWin; // so WndProc can update win->width/height on resize
};

static void __gui_win32_push(struct GuiWin32Platform* p, struct GuiEvent ev) {
    unsafe {
        int next = (p->queueTail + 1) % GUI_WIN32_QUEUE_CAP;
        if (next == p->queueHead) { return; } // full — drop (best-effort queue)
        p->queue[p->queueTail] = ev;
        p->queueTail = next;
    }
}

static int __gui_win32_keycode(unsigned long vk) {
    if (vk == 0x25UL) { return GUI_KEY_LEFT; }
    if (vk == 0x27UL) { return GUI_KEY_RIGHT; }
    if (vk == 0x26UL) { return GUI_KEY_UP; }
    if (vk == 0x28UL) { return GUI_KEY_DOWN; }
    if (vk == 0x0DUL) { return GUI_KEY_ENTER; }
    if (vk == 0x1BUL) { return GUI_KEY_ESCAPE; }
    if (vk == 0x09UL) { return GUI_KEY_TAB; }
    if (vk == 0x08UL) { return GUI_KEY_BACKSPACE; }
    if (vk == 0x20UL) { return GUI_KEY_SPACE; }
    if (vk == 0x2EUL) { return GUI_KEY_DELETE; }
    return GUI_KEY_UNKNOWN;
}

// WndProc: pulls the per-window GuiWin32Platform back out of GWLP_USERDATA
// (set right after CreateWindowExW in gui_create_window), translates the
// message into a GuiEvent, and queues it — actual application logic never
// runs inside this callback, only gui_poll_event()'s caller sees events.
void* __gui_win32_wndproc(void* hwnd, unsigned int msg, unsigned long wparam, long lparam) {
    unsafe {
        void* userData = GetWindowLongPtrW(hwnd, GUI_GWLP_USERDATA);
        if (userData == (void*)0) {
            return DefWindowProcA(hwnd, msg, wparam, lparam);
        }
        struct GuiWin32Platform* p = (struct GuiWin32Platform*)userData;

        if (msg == GUI_WM_CLOSE) {
            p->closeRequested = 1;
            return (void*)0;
        }
        if (msg == GUI_WM_DESTROY) {
            PostQuitMessage(0);
            return (void*)0;
        }
        if (msg == GUI_WM_SIZE) {
            int newW = (int)(lparam & 0xFFFF);
            int newH = (int)((lparam >> 16) & 0xFFFF);
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = GUI_EVENT_RESIZE; ev.width = newW; ev.height = newH;
            ev.x = 0.0; ev.y = 0.0; ev.button = 0; ev.keycode = 0; ev.codepoint = 0U;
            ev.scrollDx = 0.0; ev.scrollDy = 0.0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        if (msg == GUI_WM_LBUTTONDOWN || msg == GUI_WM_RBUTTONDOWN) {
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = GUI_EVENT_MOUSE_DOWN;
            ev.x = (double)(lparam & 0xFFFF); ev.y = (double)((lparam >> 16) & 0xFFFF);
            ev.button = (msg == GUI_WM_LBUTTONDOWN) ? 0 : 1;
            ev.keycode = 0; ev.codepoint = 0U; ev.scrollDx = 0.0; ev.scrollDy = 0.0;
            ev.width = 0; ev.height = 0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        if (msg == GUI_WM_LBUTTONUP || msg == GUI_WM_RBUTTONUP) {
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = GUI_EVENT_MOUSE_UP;
            ev.x = (double)(lparam & 0xFFFF); ev.y = (double)((lparam >> 16) & 0xFFFF);
            ev.button = (msg == GUI_WM_LBUTTONUP) ? 0 : 1;
            ev.keycode = 0; ev.codepoint = 0U; ev.scrollDx = 0.0; ev.scrollDy = 0.0;
            ev.width = 0; ev.height = 0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        if (msg == GUI_WM_MOUSEMOVE) {
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = GUI_EVENT_MOUSE_MOVE;
            ev.x = (double)(lparam & 0xFFFF); ev.y = (double)((lparam >> 16) & 0xFFFF);
            ev.button = 0; ev.keycode = 0; ev.codepoint = 0U; ev.scrollDx = 0.0; ev.scrollDy = 0.0;
            ev.width = 0; ev.height = 0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        if (msg == GUI_WM_KEYDOWN || msg == GUI_WM_KEYUP) {
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = (msg == GUI_WM_KEYDOWN) ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP;
            ev.keycode = __gui_win32_keycode(wparam);
            // A printable character arrives separately via WM_CHAR in a
            // full implementation; not wired up here (see file header —
            // untested backend, keycode-only text input for now).
            ev.codepoint = 0U;
            ev.x = 0.0; ev.y = 0.0; ev.button = 0; ev.scrollDx = 0.0; ev.scrollDy = 0.0;
            ev.width = 0; ev.height = 0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        if (msg == GUI_WM_MOUSEWHEEL) {
            struct GuiEvent ev;
            ev.defaultPrevented = 0;
            ev.kind = GUI_EVENT_SCROLL;
            long delta = (lparam >> 16) & 0xFFFF;
            ev.scrollDy = (double)delta / 120.0;
            ev.scrollDx = 0.0;
            ev.x = 0.0; ev.y = 0.0; ev.button = 0; ev.keycode = 0; ev.codepoint = 0U;
            ev.width = 0; ev.height = 0;
            __gui_win32_push(p, ev);
            return (void*)0;
        }
        return DefWindowProcA(hwnd, msg, wparam, lparam);
    }
}

struct GuiWindow gui_create_window(const char* title, int width, int height) {
    struct GuiWindow win;
    win.platform = (void*)0;
    win.width = width;
    win.height = height;
    win.shouldClose = 0;

    unsigned long npix = (unsigned long)width * (unsigned long)height * 4UL;
    unsafe {
        win.pixels = (&heap unsigned char)alloc(npix);
        memset((void*)win.pixels, 0, npix);
    }

    unsafe {
        struct GuiWin32Platform* p = (struct GuiWin32Platform*)alloc((unsigned long)sizeof(struct GuiWin32Platform));
        p->hwnd = (void*)0;
        p->queueHead = 0;
        p->queueTail = 0;
        p->closeRequested = 0;
        p->selfWin = (struct GuiWindow*)0;

        void* inst = GetModuleHandleA((const void*)0);

        struct Win32WndClassExA wc;
        wc.cbSize = (unsigned int)sizeof(struct Win32WndClassExA);
        wc.style = 0U;
        wc.lpfnWndProc = (void*)__gui_win32_wndproc;
        wc.cbClsExtra = 0;
        wc.cbWndExtra = 0;
        wc.hInstance = inst;
        wc.hIcon = (void*)0;
        wc.hCursor = LoadCursorA((void*)0, (const void*)32512); // IDC_ARROW
        wc.hbrBackground = (void*)0;
        wc.lpszMenuName = (const char*)0;
        wc.lpszClassName = "SafeCGuiWindowClass";
        wc.hIconSm = (void*)0;
        RegisterClassExA((const void*)&wc);

        // WS_OVERLAPPEDWINDOW (0x00CF0000) | WS_VISIBLE (0x10000000)
        void* hwnd = CreateWindowExA(0U, "SafeCGuiWindowClass", title, 0x10CF0000U,
                                      100, 100, width, height,
                                      (void*)0, (void*)0, inst, (void*)0);
        p->hwnd = hwnd;
        if (hwnd != (void*)0) {
            SetWindowLongPtrW(hwnd, GUI_GWLP_USERDATA, (void*)p);
            ShowWindow(hwnd, 1 /* SW_SHOWNORMAL */);
            UpdateWindow(hwnd);
            win.platform = (void*)p;
        } else {
            win.platform = (void*)0;
        }
    }
    return win;
}

void gui_present(struct GuiWindow* win) {
    int hasPlatform = 0;
    unsafe { if (win->platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform) { return; }
    unsafe {
        struct GuiWin32Platform* p = (struct GuiWin32Platform*)win->platform;
        void* dc = GetDC(p->hwnd);
        // BITMAPINFOHEADER, biHeight negative = top-down DIB (matches our
        // top-left-origin row-major buffer directly, no vertical flip).
        int bmi[10];
        bmi[0] = 40;                 // biSize
        bmi[1] = win->width;         // biWidth
        bmi[2] = -win->height;       // biHeight (top-down)
        bmi[3] = (1 << 16) | 32;     // biPlanes=1 (low word), biBitCount=32 (high word)
        bmi[4] = 0;                  // biCompression = BI_RGB
        bmi[5] = 0; bmi[6] = 0; bmi[7] = 0; bmi[8] = 0; bmi[9] = 0;
        StretchDIBits(dc, 0, 0, win->width, win->height, 0, 0, win->width, win->height,
                       (const void*)win->pixels, (const void*)bmi, 0U, 0x00CC0020U /* SRCCOPY */);
        ReleaseDC(p->hwnd, dc);
    }
}

void gui_set_pixel(struct GuiWindow* win, int x, int y, unsigned int rgba) {
    int oob = 0;
    unsafe { if (x < 0 || y < 0 || x >= win->width || y >= win->height) { oob = 1; } }
    if (oob) { return; }
    unsafe {
        unsigned long idx = ((unsigned long)y * (unsigned long)win->width + (unsigned long)x) * 4UL;
        // Win32's DIB is BGRA byte order for BI_RGB 32bpp, unlike Cocoa's
        // RGBA CGImage — swap R/B here so gui_draw.h's gui_rgba(r,g,b,a)
        // still means the same visual color on both backends.
        win->pixels[idx + 0UL] = (unsigned char)((rgba >> 8)  & 0xFFU); // B
        win->pixels[idx + 1UL] = (unsigned char)((rgba >> 16) & 0xFFU); // G
        win->pixels[idx + 2UL] = (unsigned char)((rgba >> 24) & 0xFFU); // R
        win->pixels[idx + 3UL] = (unsigned char)(rgba & 0xFFU);          // A
    }
}

int gui_poll_event(struct GuiWindow* win, struct GuiEvent* outEvent) {
    unsafe { outEvent->kind = GUI_EVENT_NONE; outEvent->defaultPrevented = 0; }
    int hasPlatform = 0;
    unsafe { if (win->platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform) { return 0; }

    unsafe {
        struct GuiWin32Platform* p = (struct GuiWin32Platform*)win->platform;

        // Pump the Win32 message loop (non-blocking) so WndProc runs and
        // fills the queue.
        struct Win32Msg msg;
        while (PeekMessageA((void*)&msg, (void*)0, 0U, 0U, 1U)) {
            TranslateMessage((const void*)&msg);
            DispatchMessageA((const void*)&msg);
        }

        if (p->closeRequested) {
            p->closeRequested = 0;
            win->shouldClose = 1;
            outEvent->kind = GUI_EVENT_CLOSE;
            return 1;
        }

        if (p->queueHead == p->queueTail) { return 0; }
        *outEvent = p->queue[p->queueHead];
        p->queueHead = (p->queueHead + 1) % GUI_WIN32_QUEUE_CAP;
        return 1;
    }
}

void gui_destroy_window(struct GuiWindow* win) {
    unsafe {
        if (win->platform != (void*)0) {
            struct GuiWin32Platform* p = (struct GuiWin32Platform*)win->platform;
            if (p->hwnd != (void*)0) { DestroyWindow(p->hwnd); }
            dealloc((void*)p);
        }
        if ((void*)win->pixels != (void*)0) { dealloc((void*)win->pixels); }
        win->platform = (void*)0;
    }
}

} // namespace std
