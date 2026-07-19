// SafeC Standard Library — GUI backend: Linux/BSD/Unix (X11 via Xlib). See
// gui.h for the portable API this implements.
//
// NOTE ON VERIFICATION: like gui_win32.sc, this backend is written
// carefully against the real Xlib API/ABI (XOpenDisplay, XCreateSimpleWindow,
// XSelectInput, the XEvent union's per-type struct layouts, XPutImage for
// presenting) but could not be linked or executed in the environment this
// was developed in (no X server / Xlib headers available — a real Mac,
// not Linux, and no XQuartz installed). It did get the same real-
// compiler type/syntax check every other file here did (safec's front
// end doesn't need the actual libX11 to type-check code against `extern`
// declarations) — sanity-check it end-to-end on a real X11 host before
// depending on it.
#pragma once
#include <std/gui/gui.h>
#include <std/mem.sc>

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
};

static int __gui_x11_keycode(unsigned int kc) {
    // Common X11 keycodes on a typical PC-101 layout under Xorg.
    if (kc == 113U) { return GUI_KEY_LEFT; }
    if (kc == 114U) { return GUI_KEY_RIGHT; }
    if (kc == 111U) { return GUI_KEY_UP; }
    if (kc == 116U) { return GUI_KEY_DOWN; }
    if (kc == 36U)  { return GUI_KEY_ENTER; }
    if (kc == 9U)   { return GUI_KEY_ESCAPE; }
    if (kc == 23U)  { return GUI_KEY_TAB; }
    if (kc == 22U)  { return GUI_KEY_BACKSPACE; }
    if (kc == 65U)  { return GUI_KEY_SPACE; }
    if (kc == 119U) { return GUI_KEY_DELETE; }
    return GUI_KEY_UNKNOWN;
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
        void* display = XOpenDisplay((const char*)0);
        if (display == (void*)0) {
            win.platform = (void*)0;
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

        struct GuiX11Platform* p = (struct GuiX11Platform*)alloc((unsigned long)sizeof(struct GuiX11Platform));
        p->display = display;
        p->window = window;
        p->gc = gc;
        p->wmDeleteMessage = wmDelete;
        p->closeRequested = 0;
        (void)screen;

        win.platform = (void*)p;
    }
    return win;
}

void gui_present(struct GuiWindow* win) {
    int hasPlatform = 0;
    unsafe { if (win->platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform) { return; }
    unsafe {
        struct GuiX11Platform* p = (struct GuiX11Platform*)win->platform;
        int screen = XDefaultScreen(p->display);
        void* visual = XDefaultVisual(p->display, screen);
        unsigned int depth = XDefaultDepth(p->display, screen);
        // ZPixmap = 2. bytesPerLine=0 lets Xlib compute it from width*32bpp.
        void* image = XCreateImage(p->display, visual, depth, 2, 0,
                                    (const void*)win->pixels,
                                    (unsigned int)win->width, (unsigned int)win->height,
                                    32, 0);
        if (image != (void*)0) {
            XPutImage(p->display, p->window, p->gc, image, 0, 0, 0, 0,
                      (unsigned int)win->width, (unsigned int)win->height);
            // XCreateImage takes ownership of 'data' for XDestroyImage's
            // sake in real Xlib usage; since win->pixels is *our* buffer
            // (reused every frame, not handed off), a real integration
            // must NOT let XDestroyImage free it — build with a static
            // XImage and memcpy per frame instead of XCreateImage-per-
            // frame if profiling ever shows this path matters; unverified
            // either way (see file header).
        }
        XFlush(p->display);
    }
}

void gui_set_pixel(struct GuiWindow* win, int x, int y, unsigned int rgba) {
    int oob = 0;
    unsafe { if (x < 0 || y < 0 || x >= win->width || y >= win->height) { oob = 1; } }
    if (oob) { return; }
    unsafe {
        unsigned long idx = ((unsigned long)y * (unsigned long)win->width + (unsigned long)x) * 4UL;
        // XCreateImage with ZPixmap defaults to the server's native byte
        // order for 32bpp TrueColor, virtually always BGRA/little-endian
        // on X86/ARM Linux (matches Win32's DIB convention) — swap R/B
        // here for the same reason gui_win32.sc does.
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
        struct GuiX11Platform* p = (struct GuiX11Platform*)win->platform;
        if (XPending(p->display) == 0) { return 0; }

        struct X11EventBuf buf;
        XNextEvent(p->display, (void*)&buf);
        struct X11AnyEvent* any = (struct X11AnyEvent*)&buf;
        int t = any->type;

        if (t == GUI_X11_CLIENT_MESSAGE_TYPE) {
            struct X11ClientMessageEvent* cm = (struct X11ClientMessageEvent*)&buf;
            if ((unsigned long)cm->data0 == p->wmDeleteMessage) {
                win->shouldClose = 1;
                outEvent->kind = GUI_EVENT_CLOSE;
                return 1;
            }
            return 0;
        }
        if (t == GUI_X11_BUTTON_PRESS_TYPE || t == GUI_X11_BUTTON_RELEASE_TYPE) {
            struct X11ButtonEvent* be = (struct X11ButtonEvent*)&buf;
            outEvent->kind = (t == GUI_X11_BUTTON_PRESS_TYPE) ? GUI_EVENT_MOUSE_DOWN : GUI_EVENT_MOUSE_UP;
            outEvent->x = (double)be->x; outEvent->y = (double)be->y;
            // Button1=left(1), Button2=middle, Button3=right in X11's
            // 1-based numbering — map to gui.h's 0=left/1=right/2=other.
            if (be->button == 1U) { outEvent->button = 0; }
            else if (be->button == 3U) { outEvent->button = 1; }
            else { outEvent->button = 2; }
            return 1;
        }
        if (t == GUI_X11_MOTION_NOTIFY_TYPE) {
            struct X11MotionEvent* me = (struct X11MotionEvent*)&buf;
            outEvent->kind = GUI_EVENT_MOUSE_MOVE;
            outEvent->x = (double)me->x; outEvent->y = (double)me->y;
            return 1;
        }
        if (t == GUI_X11_KEY_PRESS_TYPE || t == GUI_X11_KEY_RELEASE_TYPE) {
            struct X11KeyEvent* ke = (struct X11KeyEvent*)&buf;
            outEvent->kind = (t == GUI_X11_KEY_PRESS_TYPE) ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP;
            outEvent->keycode = __gui_x11_keycode(ke->keycode);
            // Full text (via XLookupString + an XIC for proper input-
            // method support) isn't wired up here — see file header;
            // codepoint stays 0, same keycode-only limitation gui_win32.sc
            // documents.
            outEvent->codepoint = 0U;
            return 1;
        }
        if (t == GUI_X11_CONFIGURE_NOTIFY_TYPE) {
            struct X11ConfigureEvent* ce = (struct X11ConfigureEvent*)&buf;
            outEvent->kind = GUI_EVENT_RESIZE;
            outEvent->width = ce->w; outEvent->height = ce->h;
            return 1;
        }
        return 0;
    }
}

void gui_destroy_window(struct GuiWindow* win) {
    unsafe {
        if (win->platform != (void*)0) {
            struct GuiX11Platform* p = (struct GuiX11Platform*)win->platform;
            if (p->gc != (void*)0) { XFreeGC(p->display, p->gc); }
            XDestroyWindow(p->display, p->window);
            XCloseDisplay(p->display);
            dealloc((void*)p);
        }
        if ((void*)win->pixels != (void*)0) { dealloc((void*)win->pixels); }
        win->platform = (void*)0;
    }
}

} // namespace std
