// SafeC Standard Library — GUI backend: Windows (Win32 API). The struct
// GuiBackend for Win32 (see gui_backend.h for the contract it implements
// and gui.h for the portable API on top).
//
// Verified under Wine (tests/stdlib/win32/run.sh): client size, pixels
// read back with GetPixel, and key, text, mouse, wheel, resize, move,
// focus and close messages for two windows, plus the clipboard. Not yet
// run on a real Windows host.
//
// Event model translation: Win32 delivers input via window-procedure
// callbacks (WndProc), not a poll-a-queue-yourself model like Cocoa/X11 —
// so WndProc here just translates each message into a GuiEvent posted to
// that window's portable queue (gui_backend_post), and pump() runs the
// PeekMessage/DispatchMessage loop that makes WndProc get called (waiting
// in MsgWaitForMultipleObjects when asked to block).
//
// Text: the 'W' (UTF-16) API throughout — titles and the clipboard are
// converted from/to UTF-8, WM_CHAR's UTF-16 units (surrogate pairs too)
// become codepoints. DPI: the process is made DPI-aware
// (SetProcessDPIAware) unless $SAFEC_GUI_DPI_UNAWARE is set; hiDpi windows
// then get scale = the display's DPI / 96, other windows scale 1.
// safeguard: link windows -luser32 -lgdi32
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_headless.h>
#include <std/mem.h>
#include <std/time.h>
#include <std/collections/string.h>
extern char* getenv(const char* name);

namespace std {

// Opaque Win32 handles: HWND, HINSTANCE/HMODULE, HCURSOR, HICON, HBRUSH,
// HMENU, HDC, HGLOBAL, and a kernel object HANDLE (never passed).
struct HWND_      { char opaque_; };
struct HINSTANCE_ { char opaque_; };
struct HCURSOR_   { char opaque_; };
struct HICON_     { char opaque_; };
struct HBRUSH_    { char opaque_; };
struct HMENU_     { char opaque_; };
struct HDC_       { char opaque_; };
struct HGLOBAL_   { char opaque_; };
struct WinHandle_ { char opaque_; };
struct WinCreateParam_ { char opaque_; };   // CreateWindowExW's lpParam (unused)

// WNDPROC.
typedef fn long long(HWND_*, unsigned int, unsigned long long, long long) Win32WndProc_;

// Mirrors WNDCLASSEXW/MSG's real field layout directly (rather than a
// hand-packed byte/word array) so standard C struct alignment rules give
// an ABI-compatible layout without manual offset arithmetic. WPARAM and
// LPARAM are 64-bit on Win64 (Windows' own 'long' is 32-bit).
struct Win32WndClassExW {
    unsigned int cbSize;
    unsigned int style;
    Win32WndProc_ lpfnWndProc;
    int cbClsExtra;
    int cbWndExtra;
    HINSTANCE_* hInstance;
    HICON_* hIcon;
    HCURSOR_* hCursor;
    HBRUSH_* hbrBackground;
    const unsigned short* lpszMenuName;
    const unsigned short* lpszClassName;
    HICON_* hIconSm;
};

struct Win32Msg {
    HWND_* hwnd;
    unsigned int message;
    unsigned long long wParam;
    long long lParam;
    unsigned int time;
    int ptX;
    int ptY;
    unsigned int lPrivate;
};

struct Win32Rect { int left; int top; int right; int bottom; };
struct Win32Point { int x; int y; };
struct Win32MinMaxInfo {
    struct Win32Point reserved;
    struct Win32Point maxSize;
    struct Win32Point maxPosition;
    struct Win32Point minTrackSize;
    struct Win32Point maxTrackSize;
};
struct Win32TrackMouseEvent {
    unsigned int cbSize;
    unsigned int dwFlags;
    HWND_* hwndTrack;
    unsigned int dwHoverTime;
};
struct Win32BitmapInfoHeader {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};

extern HINSTANCE_* GetModuleHandleW(const unsigned short* moduleName);
extern unsigned short RegisterClassExW(const Win32WndClassExW* wndClassEx);
extern HWND_* CreateWindowExW(unsigned int exStyle, const unsigned short* className, const unsigned short* windowName,
                              unsigned int style, int x, int y, int w, int h,
                              HWND_* parent, HMENU_* menu, HINSTANCE_* instance, WinCreateParam_* param);
extern int ShowWindow(HWND_* hwnd, int cmdShow);
extern int UpdateWindow(HWND_* hwnd);
extern int PeekMessageW(Win32Msg* msg, HWND_* hwnd, unsigned int msgMin, unsigned int msgMax, unsigned int remove);
extern int TranslateMessage(const Win32Msg* msg);
extern long long DispatchMessageW(const Win32Msg* msg);
extern long long DefWindowProcW(HWND_* hwnd, unsigned int msg, unsigned long long wparam, long long lparam);
extern HCURSOR_* LoadCursorW(HINSTANCE_* instance, const unsigned short* cursorName);
extern HCURSOR_* SetCursor(HCURSOR_* cursor);
extern HDC_* GetDC(HWND_* hwnd);
extern int ReleaseDC(HWND_* hwnd, HDC_* dc);
extern int GetDeviceCaps(HDC_* dc, int index);
extern int StretchDIBits(HDC_* dc, int destX, int destY, int destW, int destH,
                         int srcX, int srcY, int srcW, int srcH,
                         const unsigned char* bits, const Win32BitmapInfoHeader* bitmapInfo,
                         unsigned int usage, unsigned int rop);
extern int DestroyWindow(HWND_* hwnd);
extern long long SetWindowLongPtrW(HWND_* hwnd, int index, long long newLong);
extern long long GetWindowLongPtrW(HWND_* hwnd, int index);
extern int AdjustWindowRectEx(Win32Rect* rect, unsigned int style, int menu, unsigned int exStyle);
extern int SetWindowTextW(HWND_* hwnd, const unsigned short* text);
extern int SetWindowPos(HWND_* hwnd, HWND_* after, int x, int y, int cx, int cy, unsigned int flags);
extern int SetForegroundWindow(HWND_* hwnd);
extern HWND_* SetFocus(HWND_* hwnd);
extern HWND_* SetCapture(HWND_* hwnd);
extern int ReleaseCapture();
extern int TrackMouseEvent(Win32TrackMouseEvent* tme);
extern int ScreenToClient(HWND_* hwnd, Win32Point* point);
extern int ClientToScreen(HWND_* hwnd, Win32Point* point);
extern short GetKeyState(int vk);
extern int ValidateRect(HWND_* hwnd, const Win32Rect* rect);
extern unsigned int MsgWaitForMultipleObjects(unsigned int count, WinHandle_* const* handles, int waitAll,
                                              unsigned int ms, unsigned int wakeMask);
extern int PostThreadMessageW(unsigned int thread, unsigned int msg, unsigned long long wparam, long long lparam);
extern unsigned int GetCurrentThreadId();
extern int SetProcessDPIAware();
extern int OpenClipboard(HWND_* hwnd);
extern int CloseClipboard();
extern int EmptyClipboard();
extern HGLOBAL_* SetClipboardData(unsigned int format, HGLOBAL_* mem);
extern HGLOBAL_* GetClipboardData(unsigned int format);
extern HGLOBAL_* GlobalAlloc(unsigned int flags, unsigned long long bytes);
extern unsigned short* GlobalLock(HGLOBAL_* mem);   // the CF_UNICODETEXT text here
extern int GlobalUnlock(HGLOBAL_* mem);
extern HGLOBAL_* GlobalFree(HGLOBAL_* mem);

#define GUI_WM_NULL          0x0000U
#define GUI_WM_DESTROY       0x0002U
#define GUI_WM_MOVE          0x0003U
#define GUI_WM_SIZE          0x0005U
#define GUI_WM_SETFOCUS      0x0007U
#define GUI_WM_KILLFOCUS     0x0008U
#define GUI_WM_PAINT         0x000FU
#define GUI_WM_CLOSE         0x0010U
#define GUI_WM_ERASEBKGND    0x0014U
#define GUI_WM_SETCURSOR     0x0020U
#define GUI_WM_GETMINMAXINFO 0x0024U
#define GUI_WM_KEYDOWN       0x0100U
#define GUI_WM_KEYUP         0x0101U
#define GUI_WM_CHAR          0x0102U
#define GUI_WM_SYSKEYDOWN    0x0104U
#define GUI_WM_SYSKEYUP      0x0105U
#define GUI_WM_SYSCHAR       0x0106U
#define GUI_WM_MOUSEMOVE     0x0200U
#define GUI_WM_LBUTTONDOWN   0x0201U
#define GUI_WM_LBUTTONUP     0x0202U
#define GUI_WM_LBUTTONDBLCLK 0x0203U
#define GUI_WM_RBUTTONDOWN   0x0204U
#define GUI_WM_RBUTTONUP     0x0205U
#define GUI_WM_MBUTTONDOWN   0x0207U
#define GUI_WM_MBUTTONUP     0x0208U
#define GUI_WM_MOUSEWHEEL    0x020AU
#define GUI_WM_MOUSEHWHEEL   0x020EU
#define GUI_WM_MOUSELEAVE    0x02A3U
#define GUI_WM_DPICHANGED    0x02E0U
#define GUI_WM_USER          0x0400U
#define GUI_GWLP_USERDATA    (-21)

#define GUI_WS_OVERLAPPEDWINDOW 0x00CF0000U
#define GUI_WS_FIXED            0x00CA0000U   // caption | sysmenu | minimize box
#define GUI_WS_POPUP            0x80000000U
#define GUI_WS_EX_TOPMOST       0x00000008U
#define GUI_CW_USEDEFAULT       ((int)0x80000000U)

struct GuiWin32Window {
    HWND_* hwnd;
    unsigned int style;
    unsigned int exStyle;
    unsigned char* staging;      // BGRA copy of the buffer
    int stagingW;
    int stagingH;
    int tracking;                // TrackMouseEvent armed (pointer inside)
    unsigned int highSurrogate;  // first half of a WM_CHAR surrogate pair
    int buttonsDown;
};

static struct GuiBackend gW32Backend;
static int gW32BackendInit = 0;
static HINSTANCE_* gW32Instance = (HINSTANCE_*)0;
static unsigned int gW32Thread = 0U;
static int gW32ClassRegistered = 0;
static double gW32Scale = 1.0;
static unsigned short gW32ClassName[20];
static HCURSOR_* gW32Cursors[GUI_CURSOR_COUNT];

static struct GuiWin32Window* __gui_win32_win(&GuiWindowState st) {
    unsafe { return (struct GuiWin32Window*)st.native; }
}

// UTF-8 to a NUL-terminated UTF-16 heap buffer (free with dealloc).
static unsigned short* __gui_win32_wide(const &char s) {
    unsafe {
        unsigned long n = 0UL;
        while (s[n] != '\0') { n = n + 1UL; }
        unsigned short* w = (unsigned short*)alloc_raw_((n * 2UL + 1UL) * 2UL);
        unsigned long i = 0UL;
        unsigned long o = 0UL;
        while (i < n) {
            unsigned int cp = gui_utf8_next_n_(s, n, &i);
            if (cp >= 0x10000U) {
                cp = cp - 0x10000U;
                w[o] = (unsigned short)(0xD800U + (cp >> 10));
                w[o + 1UL] = (unsigned short)(0xDC00U + (cp & 0x3FFU));
                o = o + 2UL;
            } else {
                w[o] = (unsigned short)cp;
                o = o + 1UL;
            }
        }
        w[o] = (unsigned short)0;
        return w;
    }
}

static double __gui_win32_lo(long long v) { return (double)(short)(v & 0xFFFFLL); }
static double __gui_win32_hi(long long v) { return (double)(short)((v >> 16) & 0xFFFFLL); }

static int __gui_win32_mods() {
    int m = 0;
    unsafe {
        if ((GetKeyState(0x10) & (short)0x8000) != (short)0) { m = m | GUI_MOD_SHIFT; }
        if ((GetKeyState(0x11) & (short)0x8000) != (short)0) { m = m | GUI_MOD_CTRL; }
        if ((GetKeyState(0x12) & (short)0x8000) != (short)0) { m = m | GUI_MOD_ALT; }
        if ((GetKeyState(0x5B) & (short)0x8000) != (short)0 ||
            (GetKeyState(0x5C) & (short)0x8000) != (short)0) { m = m | GUI_MOD_META; }
    }
    return m;
}

static int __gui_win32_keycode(unsigned long long vk) {
    if (vk == 0x25ULL) { return GUI_KEY_LEFT; }
    if (vk == 0x27ULL) { return GUI_KEY_RIGHT; }
    if (vk == 0x26ULL) { return GUI_KEY_UP; }
    if (vk == 0x28ULL) { return GUI_KEY_DOWN; }
    if (vk == 0x0DULL) { return GUI_KEY_ENTER; }
    if (vk == 0x1BULL) { return GUI_KEY_ESCAPE; }
    if (vk == 0x09ULL) { return GUI_KEY_TAB; }
    if (vk == 0x08ULL) { return GUI_KEY_BACKSPACE; }
    if (vk == 0x20ULL) { return GUI_KEY_SPACE; }
    if (vk == 0x2EULL) { return GUI_KEY_DELETE; }
    if (vk == 0x24ULL) { return GUI_KEY_HOME; }
    if (vk == 0x23ULL) { return GUI_KEY_END; }
    if (vk == 0x21ULL) { return GUI_KEY_PAGE_UP; }
    if (vk == 0x22ULL) { return GUI_KEY_PAGE_DOWN; }
    if (vk == 0x2DULL) { return GUI_KEY_INSERT; }
    if (vk >= 0x70ULL && vk <= 0x7BULL) { return GUI_KEY_F1 + (int)(vk - 0x70ULL); }
    if (vk == 0x10ULL || vk == 0xA0ULL || vk == 0xA1ULL) { return GUI_KEY_SHIFT; }
    if (vk == 0x11ULL || vk == 0xA2ULL || vk == 0xA3ULL) { return GUI_KEY_CONTROL; }
    if (vk == 0x12ULL || vk == 0xA4ULL || vk == 0xA5ULL) { return GUI_KEY_ALT; }
    if (vk == 0x5BULL || vk == 0x5CULL) { return GUI_KEY_META; }
    if (vk == 0x5DULL) { return GUI_KEY_MENU; }
    if (vk >= 0x41ULL && vk <= 0x5AULL) { return (int)vk; }
    if (vk >= 0x30ULL && vk <= 0x39ULL) { return (int)vk; }
    if (vk >= 0x60ULL && vk <= 0x69ULL) { return (int)(vk - 0x60ULL + 0x30ULL); }   // numpad digits
    return GUI_KEY_UNKNOWN;
}

static void __gui_win32_post_mouse(&GuiWindowState st, int kind, long long lparam, int button) {
    struct GuiEvent ev = GuiEvent::new(kind);
    ev.x = __gui_win32_lo(lparam);
    ev.y = __gui_win32_hi(lparam);
    ev.button = button;
    ev.modifiers = __gui_win32_mods();
    gui_backend_post(st, ev);
}

// The outer window size for a client area of cw x ch pixels.
static void __gui_win32_outer(struct GuiWin32Window* w, int cw, int ch, &int ow, &int oh) {
    unsafe {
        struct Win32Rect r;
        r.left = 0; r.top = 0; r.right = cw; r.bottom = ch;
        AdjustWindowRectEx(&r, w->style, 0, w->exStyle);
        *ow = r.right - r.left;
        *oh = r.bottom - r.top;
    }
}

// WndProc: pulls the window's GuiWindowState back out of GWLP_USERDATA
// (set right after CreateWindowExW), translates the message into a
// GuiEvent and posts it — application logic never runs inside this
// callback, only poll_event()'s caller sees events.
long long __gui_win32_wndproc(HWND_* hwnd, unsigned int msg, unsigned long long wparam, long long lparam) {
    unsafe {
        struct GuiWindowState* st = (struct GuiWindowState*)GetWindowLongPtrW(hwnd, GUI_GWLP_USERDATA);
        if (st == (struct GuiWindowState*)0) {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }
        struct GuiWin32Window* w = (struct GuiWin32Window*)st->native;
        if (w == (struct GuiWin32Window*)0) {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }

        if (msg == GUI_WM_CLOSE) {
            gui_backend_post(st, GuiEvent::new(GUI_EVENT_CLOSE));   // the application decides
            return 0LL;
        }
        if (msg == GUI_WM_SIZE) {
            if (wparam == 1ULL) { return 0LL; }                       // SIZE_MINIMIZED
            int newW = (int)(lparam & 0xFFFFLL);
            int newH = (int)((lparam >> 16) & 0xFFFFLL);
            if (newW < 1 || newH < 1) { return 0LL; }
            st->pixelW = newW;
            st->pixelH = newH;
            st->logicalW = (int)((double)newW / st->scale + 0.5);
            st->logicalH = (int)((double)newH / st->scale + 0.5);
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
            ev.width = newW;
            ev.height = newH;
            gui_backend_post(st, ev);
            return 0LL;
        }
        if (msg == GUI_WM_MOVE) {
            int lx = (int)(__gui_win32_lo(lparam) / st->scale);
            int ly = (int)(__gui_win32_hi(lparam) / st->scale);
            if (lx != st->x || ly != st->y) {
                st->x = lx;
                st->y = ly;
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOVE);
                ev.x = (double)lx;
                ev.y = (double)ly;
                gui_backend_post(st, ev);
            }
            return 0LL;
        }
        if (msg == GUI_WM_SETFOCUS || msg == GUI_WM_KILLFOCUS) {
            int focus = msg == GUI_WM_SETFOCUS ? 1 : 0;
            if (st->focused != focus) {
                st->focused = focus;
                gui_backend_post(st, GuiEvent::new(focus ? GUI_EVENT_FOCUS_IN : GUI_EVENT_FOCUS_OUT));
            }
            return 0LL;
        }
        if (msg == GUI_WM_PAINT) {
            ValidateRect(hwnd, (const Win32Rect*)0);
            gui_backend_post(st, GuiEvent::new(GUI_EVENT_EXPOSE));
            return 0LL;
        }
        if (msg == GUI_WM_ERASEBKGND) { return 1LL; }
        if (msg == GUI_WM_SETCURSOR) {
            if ((lparam & 0xFFFFLL) == 1LL) {                         // HTCLIENT
                SetCursor(st->cursor == GUI_CURSOR_HIDDEN ? (HCURSOR_*)0 : gW32Cursors[st->cursor]);
                return 1LL;
            }
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }
        if (msg == GUI_WM_GETMINMAXINFO) {
            struct Win32MinMaxInfo* mm = (struct Win32MinMaxInfo*)lparam;
            int ow = 0; int oh = 0;
            if (st->minW > 0 || st->minH > 0) {
                __gui_win32_outer(w, (int)((double)st->minW * st->scale), (int)((double)st->minH * st->scale), &ow, &oh);
                if (st->minW > 0) { mm->minTrackSize.x = ow; }
                if (st->minH > 0) { mm->minTrackSize.y = oh; }
            }
            if (st->maxW > 0 || st->maxH > 0) {
                __gui_win32_outer(w, (int)((double)st->maxW * st->scale), (int)((double)st->maxH * st->scale), &ow, &oh);
                if (st->maxW > 0) { mm->maxTrackSize.x = ow; }
                if (st->maxH > 0) { mm->maxTrackSize.y = oh; }
            }
            return 0LL;
        }
        if (msg == GUI_WM_DPICHANGED) {
            if (st->hiDpi) {
                double ns = (double)((wparam >> 16) & 0xFFFFULL) / 96.0;
                if (ns > 0.0 && ns != st->scale) {
                    st->scale = ns;
                    struct Win32Rect* r = (struct Win32Rect*)lparam;
                    SetWindowPos(hwnd, (HWND_*)0, r->left, r->top, r->right - r->left, r->bottom - r->top,
                                 0x0004U | 0x0010U);   // SWP_NOZORDER | SWP_NOACTIVATE
                    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCALE);
                    ev.width = st->pixelW;
                    ev.height = st->pixelH;
                    gui_backend_post(st, ev);
                }
            }
            return 0LL;
        }
        if (msg == GUI_WM_LBUTTONDOWN || msg == GUI_WM_RBUTTONDOWN || msg == GUI_WM_MBUTTONDOWN ||
            msg == GUI_WM_LBUTTONDBLCLK) {
            int button = msg == GUI_WM_RBUTTONDOWN ? GUI_BUTTON_RIGHT
                       : (msg == GUI_WM_MBUTTONDOWN ? GUI_BUTTON_MIDDLE : GUI_BUTTON_LEFT);
            if (w->buttonsDown == 0) { SetCapture(hwnd); }            // keep getting moves while dragging out
            w->buttonsDown = w->buttonsDown + 1;
            __gui_win32_post_mouse(st, GUI_EVENT_MOUSE_DOWN, lparam, button);
            return 0LL;
        }
        if (msg == GUI_WM_LBUTTONUP || msg == GUI_WM_RBUTTONUP || msg == GUI_WM_MBUTTONUP) {
            int button = msg == GUI_WM_RBUTTONUP ? GUI_BUTTON_RIGHT
                       : (msg == GUI_WM_MBUTTONUP ? GUI_BUTTON_MIDDLE : GUI_BUTTON_LEFT);
            if (w->buttonsDown > 0) {
                w->buttonsDown = w->buttonsDown - 1;
                if (w->buttonsDown == 0) { ReleaseCapture(); }
            }
            __gui_win32_post_mouse(st, GUI_EVENT_MOUSE_UP, lparam, button);
            return 0LL;
        }
        if (msg == GUI_WM_MOUSEMOVE) {
            if (!w->tracking) {
                struct Win32TrackMouseEvent tme;
                tme.cbSize = (unsigned int)sizeof(struct Win32TrackMouseEvent);
                tme.dwFlags = 2U;                                     // TME_LEAVE
                tme.hwndTrack = hwnd;
                tme.dwHoverTime = 0U;
                TrackMouseEvent(&tme);
                w->tracking = 1;
                __gui_win32_post_mouse(st, GUI_EVENT_MOUSE_ENTER, lparam, 0);
            }
            __gui_win32_post_mouse(st, GUI_EVENT_MOUSE_MOVE, lparam, 0);
            return 0LL;
        }
        if (msg == GUI_WM_MOUSELEAVE) {
            w->tracking = 0;
            gui_backend_post(st, GuiEvent::new(GUI_EVENT_MOUSE_LEAVE));
            return 0LL;
        }
        if (msg == GUI_WM_KEYDOWN || msg == GUI_WM_KEYUP || msg == GUI_WM_SYSKEYDOWN || msg == GUI_WM_SYSKEYUP) {
            int down = (msg == GUI_WM_KEYDOWN || msg == GUI_WM_SYSKEYDOWN) ? 1 : 0;
            struct GuiEvent ev = GuiEvent::new(down ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP);
            ev.keycode = __gui_win32_keycode(wparam);
            ev.modifiers = __gui_win32_mods();
            // The typed character arrives separately, as WM_CHAR right
            // after (TranslateMessage); it's put on this event then.
            gui_backend_post(st, ev);
            if (msg == GUI_WM_SYSKEYDOWN || msg == GUI_WM_SYSKEYUP) {
                return DefWindowProcW(hwnd, msg, wparam, lparam);    // Alt+F4, Alt+Space, ...
            }
            return 0LL;
        }
        if (msg == GUI_WM_CHAR || msg == GUI_WM_SYSCHAR) {
            unsigned int u = (unsigned int)(wparam & 0xFFFFULL);
            if (u >= 0xD800U && u <= 0xDBFFU) { w->highSurrogate = u; return 0LL; }
            unsigned int c = u;
            if (u >= 0xDC00U && u <= 0xDFFFU) {
                if (w->highSurrogate == 0U) { return 0LL; }
                c = 0x10000U + ((w->highSurrogate - 0xD800U) << 10) + (u - 0xDC00U);
                w->highSurrogate = 0U;
            }
            if (c < 32U || c == 127U) { return 0LL; }                  // control characters
            if (msg == GUI_WM_SYSCHAR) { return DefWindowProcW(hwnd, msg, wparam, lparam); }
            // Put the character on the KEY_DOWN that produced it (as the
            // other backends do), or post it as its own CHAR event.
            unsigned long qn = st->queue.len;
            if (qn > st->qhead) {
                struct GuiEvent* last = (struct GuiEvent*)st->queue.at(qn - 1UL);
                if (last->kind == GUI_EVENT_KEY_DOWN && last->codepoint == 0U &&
                    (last->modifiers & GUI_MOD_CTRL) == 0) {
                    last->codepoint = c;
                    return 0LL;
                }
            }
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_CHAR);
            ev.codepoint = c;
            gui_backend_post(st, ev);
            return 0LL;
        }
        if (msg == GUI_WM_MOUSEWHEEL || msg == GUI_WM_MOUSEHWHEEL) {
            // The wheel delta is the signed high word of wParam (WHEEL_DELTA
            // = 120 per notch); lParam holds screen coordinates.
            struct Win32Point pt;
            pt.x = (int)__gui_win32_lo(lparam);
            pt.y = (int)__gui_win32_hi(lparam);
            ScreenToClient(hwnd, &pt);
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCROLL);
            double d = (double)(short)((wparam >> 16) & 0xFFFFULL) / 120.0;
            if (msg == GUI_WM_MOUSEWHEEL) { ev.scrollDy = d; } else { ev.scrollDx = d; }
            ev.x = (double)pt.x;
            ev.y = (double)pt.y;
            ev.modifiers = __gui_win32_mods();
            gui_backend_post(st, ev);
            return 0LL;
        }
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
}

static int __gui_win32_init() {
    unsafe {
        char* unaware = getenv("SAFEC_GUI_DPI_UNAWARE");
        if (unaware == (char*)0) { SetProcessDPIAware(); }
        gW32Instance = GetModuleHandleW((const unsigned short*)0);
        gW32Thread = GetCurrentThreadId();
        const char* cls = "SafeCGuiWindow";
        int i = 0;
        while (cls[i] != '\0') { gW32ClassName[i] = (unsigned short)cls[i]; i = i + 1; }
        gW32ClassName[i] = (unsigned short)0;
        if (!gW32ClassRegistered) {
            struct Win32WndClassExW wc;
            safe_memset_obj(&wc, 0);
            wc.cbSize = (unsigned int)sizeof(struct Win32WndClassExW);
            wc.style = 0x0003U;                                   // CS_HREDRAW | CS_VREDRAW
            wc.lpfnWndProc = __gui_win32_wndproc;
            wc.hInstance = gW32Instance;
            wc.hCursor = (HCURSOR_*)0;                             // WM_SETCURSOR sets it
            wc.lpszClassName = &gW32ClassName[0];
            if (RegisterClassExW(&wc) == (unsigned short)0) { return 0; }
            gW32ClassRegistered = 1;
        }
        // IDC_* resource ids per GUI_CURSOR_*.
        int ids[GUI_CURSOR_COUNT];
        ids[GUI_CURSOR_ARROW] = 32512; ids[GUI_CURSOR_IBEAM] = 32513; ids[GUI_CURSOR_HAND] = 32649;
        ids[GUI_CURSOR_CROSSHAIR] = 32515; ids[GUI_CURSOR_RESIZE_EW] = 32644; ids[GUI_CURSOR_RESIZE_NS] = 32645;
        ids[GUI_CURSOR_RESIZE_NWSE] = 32642; ids[GUI_CURSOR_RESIZE_NESW] = 32643; ids[GUI_CURSOR_MOVE] = 32646;
        ids[GUI_CURSOR_WAIT] = 32514; ids[GUI_CURSOR_NOT_ALLOWED] = 32648; ids[GUI_CURSOR_HIDDEN] = 32512;
        i = 0;
        while (i < GUI_CURSOR_COUNT) {
            gW32Cursors[i] = LoadCursorW((HINSTANCE_*)0, (const unsigned short*)(unsigned long long)ids[i]);   // MAKEINTRESOURCE
            i = i + 1;
        }
        HDC_* dc = GetDC((HWND_*)0);
        int dpi = dc != (HDC_*)0 ? GetDeviceCaps(dc, 88 /* LOGPIXELSX */) : 96;
        if (dc != (HDC_*)0) { ReleaseDC((HWND_*)0, dc); }
        gW32Scale = dpi > 0 ? (double)dpi / 96.0 : 1.0;
        if (unaware != (char*)0) { gW32Scale = 1.0; }
        return 1;
    }
}

static int __gui_win32_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        struct GuiWin32Window* w = (struct GuiWin32Window*)alloc_zeroed_raw_(sizeof(struct GuiWin32Window));
        st.scale = opts.hiDpi ? gW32Scale : 1.0;
        st.pixelW = (int)((double)st.logicalW * st.scale + 0.5);
        st.pixelH = (int)((double)st.logicalH * st.scale + 0.5);
        w->style = !opts.decorated ? GUI_WS_POPUP : (opts.resizable ? GUI_WS_OVERLAPPEDWINDOW : GUI_WS_FIXED);
        w->exStyle = opts.topmost ? GUI_WS_EX_TOPMOST : 0U;
        // width/height are the client area; the window is that plus its frame.
        int ow = 0; int oh = 0;
        __gui_win32_outer(w, st.pixelW, st.pixelH, &ow, &oh);
        int px = GUI_CW_USEDEFAULT; int py = GUI_CW_USEDEFAULT;
        if (st.x != GUI_POS_DEFAULT && st.y != GUI_POS_DEFAULT) {
            px = (int)((double)st.x * st.scale);
            py = (int)((double)st.y * st.scale);
        }
        unsigned short* title = __gui_win32_wide((const char*)st.title.data);
        HWND_* hwnd = CreateWindowExW(w->exStyle, &gW32ClassName[0], title, w->style,
                                      px, py, ow, oh, (HWND_*)0, (HMENU_*)0, gW32Instance, (WinCreateParam_*)0);
        dealloc(title);
        if (hwnd == (HWND_*)0) {
            dealloc(w);
            return 0;
        }
        w->hwnd = hwnd;
        st.native = (?&Erased)w;
        st.key = (unsigned long long)hwnd;
        SetWindowLongPtrW(hwnd, GUI_GWLP_USERDATA, (long long)(struct GuiWindowState*)st);
        if (opts.visible) {
            ShowWindow(hwnd, 1 /* SW_SHOWNORMAL */);
            UpdateWindow(hwnd);
        }
        struct Win32Point origin;
        origin.x = 0; origin.y = 0;
        ClientToScreen(hwnd, &origin);
        st.x = (int)((double)origin.x / st.scale);
        st.y = (int)((double)origin.y / st.scale);
        return 1;
    }
}

static void __gui_win32_destroy(&GuiWindowState st) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        SetWindowLongPtrW(w->hwnd, GUI_GWLP_USERDATA, 0LL);
        DestroyWindow(w->hwnd);
        if (w->staging != (unsigned char*)0) { dealloc(w->staging); }
        dealloc(w);
        st.native = null;
    }
}

static void __gui_win32_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                                int x, int y, int rw, int rh) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        if (w->stagingW != bw || w->stagingH != bh || w->staging == (unsigned char*)0) {
            if (w->staging != (unsigned char*)0) { dealloc(w->staging); }
            w->staging = (unsigned char*)alloc_zeroed_raw_((unsigned long)bw * (unsigned long)bh * 4UL);
            w->stagingW = bw;
            w->stagingH = bh;
        }
        // Win32's DIB is BGRA byte order for BI_RGB 32bpp; ours is RGBA.
        const unsigned char* src0 = (const unsigned char*)px;
        int yy = y;
        while (yy < y + rh) {
            unsigned long row = ((unsigned long)yy * (unsigned long)bw + (unsigned long)x) * 4UL;
            const unsigned char* s = src0 + row;
            unsigned char* d = w->staging + row;
            int xx = 0;
            while (xx < rw) {
                d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; d[3] = s[3];
                s = s + 4; d = d + 4;
                xx = xx + 1;
            }
            yy = yy + 1;
        }
        // A top-down DIB of just the rows y..y+rh (so there's no question
        // which way the source y origin counts).
        struct Win32BitmapInfoHeader bmi;
        safe_memset_obj(&bmi, 0);
        bmi.biSize = 40U;
        bmi.biWidth = bw;
        bmi.biHeight = -rh;
        bmi.biPlanes = (unsigned short)1;
        bmi.biBitCount = (unsigned short)32;
        bmi.biCompression = 0U;                                    // BI_RGB
        HDC_* dc = GetDC(w->hwnd);
        StretchDIBits(dc, x, y, rw, rh, x, 0, rw, rh,
                      w->staging + (unsigned long)y * (unsigned long)bw * 4UL,
                      &bmi, 0U, 0x00CC0020U /* SRCCOPY */);
        ReleaseDC(w->hwnd, dc);
    }
}

static void __gui_win32_set_title(&GuiWindowState st, const &char title) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        unsigned short* t = __gui_win32_wide(title);
        SetWindowTextW(w->hwnd, t);
        dealloc(t);
    }
}

static void __gui_win32_set_size(&GuiWindowState st, int lw, int lh) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        int ow = 0; int oh = 0;
        __gui_win32_outer(w, (int)((double)lw * st.scale + 0.5), (int)((double)lh * st.scale + 0.5), &ow, &oh);
        SetWindowPos(w->hwnd, (HWND_*)0, 0, 0, ow, oh, 0x0002U | 0x0004U | 0x0010U);  // NOMOVE|NOZORDER|NOACTIVATE
    }
}

static void __gui_win32_set_position(&GuiWindowState st, int x, int y) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        // x/y place the client area; move the frame so that it lands there.
        struct Win32Rect r;
        r.left = 0; r.top = 0; r.right = 0; r.bottom = 0;
        AdjustWindowRectEx(&r, w->style, 0, w->exStyle);
        SetWindowPos(w->hwnd, (HWND_*)0, (int)((double)x * st.scale) + r.left, (int)((double)y * st.scale) + r.top,
                     0, 0, 0x0001U | 0x0004U | 0x0010U);              // NOSIZE|NOZORDER|NOACTIVATE
    }
}

static void __gui_win32_update_limits(&GuiWindowState st) {
    // Applied in WM_GETMINMAXINFO; nudge the window so the limits take
    // effect on the current size too.
    int lw = st.logicalW;
    int lh = st.logicalH;
    if (st.minW > 0 && lw < st.minW) { lw = st.minW; }
    if (st.minH > 0 && lh < st.minH) { lh = st.minH; }
    if (st.maxW > 0 && lw > st.maxW) { lw = st.maxW; }
    if (st.maxH > 0 && lh > st.maxH) { lh = st.maxH; }
    if (lw != st.logicalW || lh != st.logicalH) { __gui_win32_set_size(st, lw, lh); }
}

static void __gui_win32_set_visible(&GuiWindowState st, int v) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        ShowWindow(w->hwnd, v ? 5 /* SW_SHOW */ : 0 /* SW_HIDE */);
    }
}

static void __gui_win32_focus(&GuiWindowState st) {
    struct GuiWin32Window* w = __gui_win32_win(st);
    unsafe {
        if (w == (struct GuiWin32Window*)0) { return; }
        SetForegroundWindow(w->hwnd);
        SetFocus(w->hwnd);
    }
}

static void __gui_win32_set_cursor(&GuiWindowState st, int c) {
    // Takes effect on the next WM_SETCURSOR; apply now if the pointer is in.
    unsafe {
        if (st.hovered) { SetCursor(c == GUI_CURSOR_HIDDEN ? (HCURSOR_*)0 : gW32Cursors[c]); }
    }
}

static int __gui_win32_drain() {
    int any = 0;
    unsafe {
        struct Win32Msg msg;
        while (PeekMessageW(&msg, (HWND_*)0, 0U, 0U, 1U /* PM_REMOVE */)) {
            if (msg.hwnd == (HWND_*)0 && msg.message == GUI_WM_USER) { continue; }   // wake()
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            any = 1;
        }
    }
    return any;
}

static int __gui_win32_pump(int timeoutMs) {
    int got = __gui_win32_drain();
    if (gui_backend_has_events() || timeoutMs == 0) { return got; }
    unsafe {
        double start = (double)time_mono_ns() / 1000000.0;
        while (!gui_backend_has_events()) {
            unsigned int wait = 0xFFFFFFFFU;                           // INFINITE
            if (timeoutMs > 0) {
                int left = timeoutMs - (int)((double)time_mono_ns() / 1000000.0 - start);
                if (left <= 0) { break; }
                wait = (unsigned int)left;
            }
            unsigned int r = MsgWaitForMultipleObjects(0U, (WinHandle_* const*)0, 0, wait, 0x04FFU /* QS_ALLINPUT */);
            int drained = __gui_win32_drain();
            if (drained) { got = 1; }
            if (r == 0x102U /* WAIT_TIMEOUT */) { break; }
            if (!drained) { break; }                                  // woken (wake()) with nothing to do
        }
    }
    return got;
}

static void __gui_win32_wake() {
    unsafe { PostThreadMessageW(gW32Thread, GUI_WM_USER, 0ULL, 0LL); }
}

static long long __gui_win32_now_ns() { return time_mono_ns(); }

static HWND_* __gui_win32_any_hwnd() {
    ?&GuiWindowState ost = gui_backend_window_at(0UL);
    match (ost) {
        case null: return (HWND_*)0;
        case some(st): {
            struct GuiWin32Window* w = __gui_win32_win(st);
            unsafe { return w != (struct GuiWin32Window*)0 ? w->hwnd : (HWND_*)0; }
        }
    }
    return (HWND_*)0;
}

static int __gui_win32_clip_set(const &char utf8, unsigned long n) {
    unsafe {
        // UTF-8 (n bytes, not necessarily NUL-terminated) to UTF-16.
        unsigned short* buf = (unsigned short*)alloc_raw_((n * 2UL + 1UL) * 2UL);
        unsigned long i = 0UL;
        unsigned long o = 0UL;
        while (i < n) {
            unsigned int cp = gui_utf8_next_n_(utf8, n, &i);
            if (cp >= 0x10000U) {
                cp = cp - 0x10000U;
                buf[o] = (unsigned short)(0xD800U + (cp >> 10));
                buf[o + 1UL] = (unsigned short)(0xDC00U + (cp & 0x3FFU));
                o = o + 2UL;
            } else {
                buf[o] = (unsigned short)cp;
                o = o + 1UL;
            }
        }
        buf[o] = (unsigned short)0;
        int ok = 0;
        if (OpenClipboard(__gui_win32_any_hwnd())) {
            EmptyClipboard();
            HGLOBAL_* mem = GlobalAlloc(0x0002U /* GMEM_MOVEABLE */, (unsigned long long)((o + 1UL) * 2UL));
            if (mem != (HGLOBAL_*)0) {
                unsigned short* dst = GlobalLock(mem);
                unsigned long k = 0UL;
                while (k <= o) { dst[k] = buf[k]; k = k + 1UL; }
                GlobalUnlock(mem);
                if (SetClipboardData(13U /* CF_UNICODETEXT */, mem) != (HGLOBAL_*)0) { ok = 1; }
                else { GlobalFree(mem); }
            }
            CloseClipboard();
        }
        dealloc(buf);
        return ok;
    }
}

static int __gui_win32_clip_get(&String out) {
    out.clear();
    unsafe {
        if (!OpenClipboard(__gui_win32_any_hwnd())) { return 0; }
        int ok = 0;
        HGLOBAL_* mem = GetClipboardData(13U);
        if (mem != (HGLOBAL_*)0) {
            unsigned short* src = GlobalLock(mem);
            if (src != (unsigned short*)0) {
                unsigned long i = 0UL;
                while (src[i] != (unsigned short)0) {
                    unsigned int u = (unsigned int)src[i];
                    unsigned int cp = u;
                    if (u >= 0xD800U && u <= 0xDBFFU && src[i + 1UL] >= (unsigned short)0xDC00 &&
                        src[i + 1UL] <= (unsigned short)0xDFFF) {
                        cp = 0x10000U + ((u - 0xD800U) << 10) + ((unsigned int)src[i + 1UL] - 0xDC00U);
                        i = i + 1UL;
                    }
                    if (cp != 13U) {                                   // CRLF to LF
                        char enc[4];
                        int k = gui_utf8_encode(cp, &enc[0]);
                        out.push_n_(&enc[0], (unsigned long)k);
                    }
                    i = i + 1UL;
                }
                GlobalUnlock(mem);
                ok = 1;
            }
        }
        CloseClipboard();
        return ok;
    }
}

static double __gui_win32_display_scale() { return gW32Scale; }

static struct GuiNativeHandle __gui_win32_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_WIN32;
    unsafe { h.display = (unsigned long long)gW32Instance; }
    h.window = st.key;
    h.view = 0ULL;
    h.visual = 0ULL;
    return h;
}

static &GuiBackend __gui_native_backend() {
    if (!gW32BackendInit) {
        gW32Backend.name = "win32";
        gW32Backend.primaryMod = GUI_MOD_CTRL;
        gW32Backend.init = __gui_win32_init;
        gW32Backend.window_create = __gui_win32_create;
        gW32Backend.window_destroy = __gui_win32_destroy;
        gW32Backend.window_present = __gui_win32_present;
        gW32Backend.window_set_title = __gui_win32_set_title;
        gW32Backend.window_set_size = __gui_win32_set_size;
        gW32Backend.window_set_position = __gui_win32_set_position;
        gW32Backend.window_update_limits = __gui_win32_update_limits;
        gW32Backend.window_set_visible = __gui_win32_set_visible;
        gW32Backend.window_focus = __gui_win32_focus;
        gW32Backend.window_set_cursor = __gui_win32_set_cursor;
        gW32Backend.pump = __gui_win32_pump;
        gW32Backend.wake = __gui_win32_wake;
        gW32Backend.now_ns = __gui_win32_now_ns;
        gW32Backend.clipboard_set = __gui_win32_clip_set;
        gW32Backend.clipboard_get = __gui_win32_clip_get;
        gW32Backend.display_scale = __gui_win32_display_scale;
        gW32Backend.native_handle = __gui_win32_native;
        gW32BackendInit = 1;
    }
    return &gW32Backend;
}

// $SAFEC_GUI_BACKEND=headless selects the in-memory backend (gui_headless.h).
&GuiBackend gui_platform_backend() {
    if (gui_headless_requested()) { return gui_headless_backend(); }
    return __gui_native_backend();
}

} // namespace std
