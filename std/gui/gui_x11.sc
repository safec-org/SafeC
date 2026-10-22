// SafeC Standard Library — GUI backend: Linux/BSD/Unix (X11 via Xlib). See
// gui.h for the portable API this implements.
//
// Verified against a real X server (Xvfb): tests/stdlib/x11/run.sh draws,
// presents and reads the window back with XGetImage, and checks key, text,
// mouse, scroll, resize and close events sent through the server.
// safeguard: link linux -lX11
// safeguard: link bsd -L/usr/local/lib -lX11
// safeguard: deb linux libx11-6
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/mem.h>
extern void* memset(void* ptr, int val, size_t n);

namespace std {

extern void* XOpenDisplay(const char* displayName);
extern unsigned long XDefaultRootWindow(void* display);
extern int XDefaultScreen(void* display);
extern unsigned long XCreateSimpleWindow(void* display, unsigned long parent,
                                          int x, int y, unsigned int w, unsigned int h,
                                          unsigned int borderWidth, unsigned long border,
                                          unsigned long background);
extern int XSelectInput(void* display, unsigned long window, long eventMask);
extern int XMapWindow(void* display, unsigned long window);
extern int XStoreName(void* display, unsigned long window, const char* name);
extern int XPending(void* display);
extern int XNextEvent(void* display, void* eventOut);
extern int XDestroyWindow(void* display, unsigned long window);
extern int XCloseDisplay(void* display);
extern int XSetWMProtocols(void* display, unsigned long window, void* protocols, int count);
extern unsigned long XInternAtom(void* display, const char* name, int onlyIfExists);
extern void* XCreateGC(void* display, unsigned long drawable, unsigned long valueMask, void* values);
extern int XFreeGC(void* display, void* gc);
extern void* XCreateImage(void* display, void* visual, unsigned int depth, int format,
                           int offset, const void* data, unsigned int w, unsigned int h,
                           int bitmapPad, int bytesPerLine);
extern int XPutImage(void* display, unsigned long drawable, void* gc, void* image,
                      int srcX, int srcY, int destX, int destY, unsigned int w, unsigned int h);
extern void* XDefaultVisual(void* display, int screen);
extern unsigned int XDefaultDepth(void* display, int screen);
extern int XFlush(void* display);
extern int XDestroyImage(void* image);
extern int XLookupString(void* keyEvent, char* buf, int len, unsigned long* keysym, void* status);

#define GUI_X11_KEY_PRESS_MASK    (1L << 0)
#define GUI_X11_KEY_RELEASE_MASK  (1L << 1)
#define GUI_X11_BUTTON_PRESS_MASK (1L << 2)
#define GUI_X11_BUTTON_RELEASE_MASK (1L << 3)
#define GUI_X11_POINTER_MOTION_MASK (1L << 6)
#define GUI_X11_STRUCTURE_NOTIFY_MASK (1L << 17)

#define GUI_X11_KEY_PRESS_TYPE     2
#define GUI_X11_KEY_RELEASE_TYPE   3
#define GUI_X11_BUTTON_PRESS_TYPE  4
#define GUI_X11_BUTTON_RELEASE_TYPE 5
#define GUI_X11_MOTION_NOTIFY_TYPE 6
#define GUI_X11_CONFIGURE_NOTIFY_TYPE 22
#define GUI_X11_CLIENT_MESSAGE_TYPE 33

// Every real XEvent union member starts with these fields — enough to
// discriminate .type before reinterpreting the same buffer as one of the
// more specific structs below.
struct X11AnyEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
};

struct X11ButtonEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    unsigned long root;
    unsigned long subwindow;
    unsigned long time;
    int x; int y;
    int xRoot; int yRoot;
    unsigned int state;
    unsigned int button;
    int sameScreen;
};

struct X11MotionEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    unsigned long root;
    unsigned long subwindow;
    unsigned long time;
    int x; int y;
    int xRoot; int yRoot;
    unsigned int state;
    char isHint;
    int sameScreen;
};

struct X11KeyEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    unsigned long root;
    unsigned long subwindow;
    unsigned long time;
    int x; int y;
    int xRoot; int yRoot;
    unsigned int state;
    unsigned int keycode;
    int sameScreen;
};

struct X11ConfigureEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long event;
    unsigned long window;
    int x; int y;
    int w; int h;
    int borderWidth;
    unsigned long above;
    int overrideRedirect;
};

struct X11ClientMessageEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    unsigned long messageType;
    int format;
    long data0; long data1; long data2; long data3; long data4;
};

// Padded to >= a real XEvent union's size (24 'long's on a 64-bit build)
// so XNextEvent has room to write whichever variant actually arrived.
struct X11EventBuf {
    long pad[24];
};

struct GuiX11Platform {
    void* display;
    unsigned long window;
    void* gc;
    unsigned long wmDeleteMessage;
    int closeRequested;
    void* image;          // XImage over win.pixels, rebuilt on resize
};

// An XImage whose data is our pixel buffer (ZPixmap, 32 bits per pixel).
static void* __gui_x11_image(void* display, &GuiWindow win) {
    unsafe {
        int screen = XDefaultScreen(display);
        return XCreateImage(display, XDefaultVisual(display, screen), XDefaultDepth(display, screen),
                            2, 0, (const void*)win.pixels,
                            (unsigned int)win.width, (unsigned int)win.height, 32, 0);
    }
}

// XDestroyImage would free 'data' too; the pixels are ours, so detach them
// first (XImage starts with four ints, then the data pointer).
static void __gui_x11_free_image(void* image) {
    if (image == (void*)0) { return; }
    unsafe {
        *(void**)((unsigned long)image + 16UL) = (void*)0;
        XDestroyImage(image);
    }
}

// X keysyms (layout-independent, unlike raw keycodes) to GuiKey.
static int __gui_x11_keysym(unsigned long ks) {
    if (ks == 0xFF51UL) { return GUI_KEY_LEFT; }
    if (ks == 0xFF53UL) { return GUI_KEY_RIGHT; }
    if (ks == 0xFF52UL) { return GUI_KEY_UP; }
    if (ks == 0xFF54UL) { return GUI_KEY_DOWN; }
    if (ks == 0xFF0DUL || ks == 0xFF8DUL) { return GUI_KEY_ENTER; }   // Return, KP_Enter
    if (ks == 0xFF1BUL) { return GUI_KEY_ESCAPE; }
    if (ks == 0xFF09UL) { return GUI_KEY_TAB; }
    if (ks == 0xFF08UL) { return GUI_KEY_BACKSPACE; }
    if (ks == 0x20UL)   { return GUI_KEY_SPACE; }
    if (ks == 0xFFFFUL) { return GUI_KEY_DELETE; }
    return GUI_KEY_UNKNOWN;
}

struct GuiWindow GuiWindow::create(const &char title, int width, int height) {
    struct GuiWindow win;
    win.platform = null;
    win.width = width;
    win.height = height;
    win.shouldClose = 0;

    unsigned long npix = (unsigned long)width * (unsigned long)height * 4UL;
    unsafe {
        win.pixels = (&heap unsigned char)alloc_raw_(npix);
        memset((void*)win.pixels, 0, npix);
    }

    unsafe {
        void* display = XOpenDisplay((const char*)0);
        if (display == (void*)0) {
            win.platform = null;
            return win;
        }

        unsigned long root = XDefaultRootWindow(display);
        int screen = XDefaultScreen(display);
        unsigned long window = XCreateSimpleWindow(display, root, 0, 0,
                                                     (unsigned int)width, (unsigned int)height,
                                                     1UL, 0UL, 0UL);
        XStoreName(display, window, title);
        XSelectInput(display, window,
                     GUI_X11_KEY_PRESS_MASK | GUI_X11_KEY_RELEASE_MASK |
                     GUI_X11_BUTTON_PRESS_MASK | GUI_X11_BUTTON_RELEASE_MASK |
                     GUI_X11_POINTER_MOTION_MASK | GUI_X11_STRUCTURE_NOTIFY_MASK);

        unsigned long wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", 0);
        XSetWMProtocols(display, window, (void*)&wmDelete, 1);

        XMapWindow(display, window);
        void* gc = XCreateGC(display, window, 0UL, (void*)0);
        XFlush(display);

        struct GuiX11Platform* p = (struct GuiX11Platform*)alloc_raw_((unsigned long)sizeof(struct GuiX11Platform));
        p->display = display;
        p->window = window;
        p->gc = gc;
        p->wmDeleteMessage = wmDelete;
        p->closeRequested = 0;
        p->image = __gui_x11_image(display, &win);
        (void)screen;

        win.platform = (void*)p;
    }
    return win;
}

void GuiWindow::present() {
    int hasPlatform = 0;
    unsafe { if (!self.platform.is_null()) { hasPlatform = 1; } }
    if (!hasPlatform) { return; }
    unsafe {
        struct GuiX11Platform* p = (struct GuiX11Platform*)self.platform;
        if (p->image != (void*)0) {
            XPutImage(p->display, p->window, p->gc, p->image, 0, 0, 0, 0,
                      (unsigned int)self.width, (unsigned int)self.height);
        }
        XFlush(p->display);
    }
}

void GuiWindow::set_pixel(int x, int y, unsigned int rgba) {
    int oob = 0;
    unsafe { if (x < 0 || y < 0 || x >= self.width || y >= self.height) { oob = 1; } }
    if (oob) { return; }
    unsafe {
        unsigned long idx = ((unsigned long)y * (unsigned long)self.width + (unsigned long)x) * 4UL;
        // XCreateImage with ZPixmap defaults to the server's native byte
        // order for 32bpp TrueColor, virtually always BGRA/little-endian
        // on X86/ARM Linux (matches Win32's DIB convention) — swap R/B
        // here for the same reason gui_win32.sc does.
        self.pixels[idx + 0UL] = (unsigned char)((rgba >> 8)  & 0xFFU); // B
        self.pixels[idx + 1UL] = (unsigned char)((rgba >> 16) & 0xFFU); // G
        self.pixels[idx + 2UL] = (unsigned char)((rgba >> 24) & 0xFFU); // R
        self.pixels[idx + 3UL] = (unsigned char)(rgba & 0xFFU);          // A
    }
}

int GuiWindow::poll_event(&GuiEvent outEvent) {
    unsafe { outEvent.kind = GUI_EVENT_NONE; outEvent.defaultPrevented = 0; }
    int hasPlatform = 0;
    unsafe { if (!self.platform.is_null()) { hasPlatform = 1; } }
    if (!hasPlatform) { return 0; }

    unsafe {
        struct GuiX11Platform* p = (struct GuiX11Platform*)self.platform;
        if (XPending(p->display) == 0) { return 0; }

        struct X11EventBuf buf;
        XNextEvent(p->display, (void*)&buf);
        struct X11AnyEvent* any = (struct X11AnyEvent*)&buf;
        int t = any->type;

        if (t == GUI_X11_CLIENT_MESSAGE_TYPE) {
            struct X11ClientMessageEvent* cm = (struct X11ClientMessageEvent*)&buf;
            if ((unsigned long)cm->data0 == p->wmDeleteMessage) {
                self.shouldClose = 1;
                outEvent.kind = GUI_EVENT_CLOSE;
                return 1;
            }
            return 0;
        }
        if (t == GUI_X11_BUTTON_PRESS_TYPE || t == GUI_X11_BUTTON_RELEASE_TYPE) {
            struct X11ButtonEvent* be = (struct X11ButtonEvent*)&buf;
            if (be->button >= 4U && be->button <= 7U) {     // wheel: 4/5 up/down, 6/7 left/right
                if (t != GUI_X11_BUTTON_PRESS_TYPE) { return 0; }
                outEvent.kind = GUI_EVENT_SCROLL;
                outEvent.x = (double)be->x; outEvent.y = (double)be->y;
                outEvent.scrollDx = be->button == 6U ? -1.0 : (be->button == 7U ? 1.0 : 0.0);
                outEvent.scrollDy = be->button == 4U ? 1.0 : (be->button == 5U ? -1.0 : 0.0);
                return 1;
            }
            outEvent.kind = (t == GUI_X11_BUTTON_PRESS_TYPE) ? GUI_EVENT_MOUSE_DOWN : GUI_EVENT_MOUSE_UP;
            outEvent.x = (double)be->x; outEvent.y = (double)be->y;
            // Button1=left(1), Button2=middle, Button3=right in X11's
            // 1-based numbering — map to gui.h's 0=left/1=right/2=other.
            if (be->button == 1U) { outEvent.button = 0; }
            else if (be->button == 3U) { outEvent.button = 1; }
            else { outEvent.button = 2; }
            return 1;
        }
        if (t == GUI_X11_MOTION_NOTIFY_TYPE) {
            struct X11MotionEvent* me = (struct X11MotionEvent*)&buf;
            outEvent.kind = GUI_EVENT_MOUSE_MOVE;
            outEvent.x = (double)me->x; outEvent.y = (double)me->y;
            return 1;
        }
        if (t == GUI_X11_KEY_PRESS_TYPE || t == GUI_X11_KEY_RELEASE_TYPE) {
            outEvent.kind = (t == GUI_X11_KEY_PRESS_TYPE) ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP;
            char text[8];
            unsigned long keysym = 0UL;
            int n = XLookupString((void*)&buf, (char*)text, 8, &keysym, (void*)0);
            outEvent.keycode = __gui_x11_keysym(keysym);
            // Like the Cocoa backend: a key press also carries the typed
            // character (ASCII; no input-method composition) in codepoint.
            outEvent.codepoint = 0U;
            if (t == GUI_X11_KEY_PRESS_TYPE && n > 0) {
                unsigned int c = (unsigned int)(unsigned char)text[0];
                if (c >= 32U && c < 127U) { outEvent.codepoint = c; }
            }
            return 1;
        }
        if (t == GUI_X11_CONFIGURE_NOTIFY_TYPE) {
            struct X11ConfigureEvent* ce = (struct X11ConfigureEvent*)&buf;
            if (ce->w == self.width && ce->h == self.height) { return 0; }   // a move
            // New size: a new zeroed pixel buffer and XImage to match.
            unsigned long npix = (unsigned long)ce->w * (unsigned long)ce->h * 4UL;
            void* px = alloc_raw_(npix);
            if (px == (void*)0) { return 0; }
            memset(px, 0, npix);
            __gui_x11_free_image(p->image);
            dealloc_raw_((void*)self.pixels);
            self.pixels = (&heap unsigned char)px;
            self.width = ce->w;
            self.height = ce->h;
            p->image = __gui_x11_image(p->display, self);
            outEvent.kind = GUI_EVENT_RESIZE;
            outEvent.width = ce->w; outEvent.height = ce->h;
            return 1;
        }
        return 0;
    }
}

void GuiWindow::destroy() {
    unsafe {
        if (!self.platform.is_null()) {
            struct GuiX11Platform* p = (struct GuiX11Platform*)self.platform;
            __gui_x11_free_image(p->image);
            if (p->gc != (void*)0) { XFreeGC(p->display, p->gc); }
            XDestroyWindow(p->display, p->window);
            XCloseDisplay(p->display);
            dealloc_raw_((void*)p);
        }
        if ((void*)self.pixels != (void*)0) { dealloc_raw_((void*)self.pixels); }
        self.platform = null;
    }
}

} // namespace std
