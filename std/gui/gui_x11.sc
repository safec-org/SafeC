// SafeC Standard Library — GUI backend: Linux/BSD/Unix (X11 via Xlib). The
// struct GuiBackend for X11 (see gui_backend.h for the contract it
// implements and gui.h for the portable API on top).
//
// One display connection serves every window; native events are routed to
// the GuiWindowState whose key is the event's Window id. Text input goes
// through an X input method (XIM, Xutf8LookupString) when one opens, else
// XLookupString + keysym-to-Unicode. The clipboard is the CLIPBOARD
// selection, owned by a hidden helper window (so it outlives every visible
// window), served as UTF8_STRING/STRING/TEXT; reading another client's
// selection waits up to a second for it to answer (INCR transfers of very
// large selections are not supported). The scale factor of hiDpi windows
// is $SAFEC_GUI_SCALE if set, else the Xft.dpi resource / 96.
//
// Verified against a real X server (Xvfb): tests/stdlib/x11/run.sh draws,
// presents and reads windows back with XGetImage, checks key, text, mouse,
// scroll, resize, focus, move and close events sent through the server to
// two windows at once, and the clipboard against another X client.
// safeguard: link linux -lX11
// safeguard: link bsd -L/usr/local/lib -lX11
// safeguard: deb linux libx11-6
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_headless.h>
#include <std/gui/gui_wayland.h>
#include <std/mem.h>
#include <std/time.h>
#include <std/collections/string.h>
extern void* memset(void* ptr, int val, size_t n);
extern void* memcpy(void* dst, const void* src, size_t n);
extern char* getenv(const char* name);
extern double atof(const char* s);
extern int pipe(int* fds);
extern long write(int fd, const void* buf, unsigned long n);
extern long read(int fd, void* buf, unsigned long n);
extern int poll(void* fds, unsigned long nfds, int timeout);
extern int fcntl(int fd, int cmd, ...);

namespace std {

extern void* XOpenDisplay(const char* displayName);
extern unsigned long XDefaultRootWindow(void* display);
extern int XDefaultScreen(void* display);
extern int XConnectionNumber(void* display);
extern unsigned long XCreateSimpleWindow(void* display, unsigned long parent,
                                          int x, int y, unsigned int w, unsigned int h,
                                          unsigned int borderWidth, unsigned long border,
                                          unsigned long background);
extern int XSelectInput(void* display, unsigned long window, long eventMask);
extern int XMapWindow(void* display, unsigned long window);
extern int XUnmapWindow(void* display, unsigned long window);
extern int XStoreName(void* display, unsigned long window, const char* name);
extern int XPending(void* display);
extern int XNextEvent(void* display, void* eventOut);
extern int XCheckTypedWindowEvent(void* display, unsigned long w, int type, void* eventOut);
extern int XDestroyWindow(void* display, unsigned long window);
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
extern int XSync(void* display, int discard);
extern int XDestroyImage(void* image);
extern int XFree(void* data);
extern int XLookupString(void* keyEvent, char* buf, int len, unsigned long* keysym, void* status);
extern unsigned long XLookupKeysym(void* keyEvent, int index);
extern int XRefreshKeyboardMapping(void* mappingEvent);
extern int XMoveWindow(void* display, unsigned long w, int x, int y);
extern int XResizeWindow(void* display, unsigned long w, unsigned int width, unsigned int height);
extern int XRaiseWindow(void* display, unsigned long w);
extern int XSetInputFocus(void* display, unsigned long w, int revertTo, unsigned long time);
extern void XSetWMNormalHints(void* display, unsigned long w, void* hints);
extern int XChangeProperty(void* display, unsigned long w, unsigned long property, unsigned long type,
                           int format, int mode, const void* data, int nelements);
extern int XGetWindowProperty(void* display, unsigned long w, unsigned long property, long offset,
                              long length, int del, unsigned long reqType, unsigned long* actualType,
                              int* actualFormat, unsigned long* nitems, unsigned long* bytesAfter,
                              unsigned char** prop);
extern int XSendEvent(void* display, unsigned long w, int propagate, long mask, void* event);
extern int XSetSelectionOwner(void* display, unsigned long selection, unsigned long owner, unsigned long time);
extern unsigned long XGetSelectionOwner(void* display, unsigned long selection);
extern int XConvertSelection(void* display, unsigned long selection, unsigned long target,
                             unsigned long property, unsigned long requestor, unsigned long time);
extern unsigned long XCreateFontCursor(void* display, unsigned int shape);
extern int XDefineCursor(void* display, unsigned long w, unsigned long cursor);
extern unsigned long XCreateBitmapFromData(void* display, unsigned long d, const char* data,
                                           unsigned int w, unsigned int h);
extern unsigned long XCreatePixmapCursor(void* display, unsigned long src, unsigned long mask,
                                         void* fg, void* bg, unsigned int x, unsigned int y);
extern int XFreePixmap(void* display, unsigned long p);
extern int XTranslateCoordinates(void* display, unsigned long src, unsigned long dst, int x, int y,
                                 int* dx, int* dy, unsigned long* child);
extern char* XResourceManagerString(void* display);
extern void* XOpenIM(void* display, void* db, char* resName, char* resClass);
extern void* XCreateIC(void* im, ...);
extern void XDestroyIC(void* ic);
extern void XSetICFocus(void* ic);
extern void XUnsetICFocus(void* ic);
extern int Xutf8LookupString(void* ic, void* keyEvent, char* buf, int len, unsigned long* keysym, int* status);
extern int XFilterEvent(void* event, unsigned long w);
extern char* XSetLocaleModifiers(const char* mods);

#define GUI_X11_KEY_PRESS_MASK       (1L << 0)
#define GUI_X11_KEY_RELEASE_MASK     (1L << 1)
#define GUI_X11_BUTTON_PRESS_MASK    (1L << 2)
#define GUI_X11_BUTTON_RELEASE_MASK  (1L << 3)
#define GUI_X11_ENTER_WINDOW_MASK    (1L << 4)
#define GUI_X11_LEAVE_WINDOW_MASK    (1L << 5)
#define GUI_X11_POINTER_MOTION_MASK  (1L << 6)
#define GUI_X11_EXPOSURE_MASK        (1L << 15)
#define GUI_X11_STRUCTURE_NOTIFY_MASK (1L << 17)
#define GUI_X11_FOCUS_CHANGE_MASK    (1L << 21)
#define GUI_X11_PROPERTY_CHANGE_MASK (1L << 22)

#define GUI_X11_KEY_PRESS_TYPE        2
#define GUI_X11_KEY_RELEASE_TYPE      3
#define GUI_X11_BUTTON_PRESS_TYPE     4
#define GUI_X11_BUTTON_RELEASE_TYPE   5
#define GUI_X11_MOTION_NOTIFY_TYPE    6
#define GUI_X11_ENTER_NOTIFY_TYPE     7
#define GUI_X11_LEAVE_NOTIFY_TYPE     8
#define GUI_X11_FOCUS_IN_TYPE         9
#define GUI_X11_FOCUS_OUT_TYPE        10
#define GUI_X11_EXPOSE_TYPE           12
#define GUI_X11_CONFIGURE_NOTIFY_TYPE 22
#define GUI_X11_SELECTION_CLEAR_TYPE  29
#define GUI_X11_SELECTION_REQUEST_TYPE 30
#define GUI_X11_SELECTION_NOTIFY_TYPE 31
#define GUI_X11_CLIENT_MESSAGE_TYPE   33
#define GUI_X11_MAPPING_NOTIFY_TYPE   34

#define GUI_X11_XA_ATOM   4UL
#define GUI_X11_XA_STRING 31UL

// Every real XEvent union member starts with these fields — enough to
// discriminate .type before reinterpreting the same buffer as one of the
// more specific structs below. Layouts are Xlib's (LP64).
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

struct X11CrossingEvent {
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
    int mode;
    int detail;
    int sameScreen;
    int focus;
    unsigned int state;
};

struct X11FocusEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    int mode;
    int detail;
};

struct X11ExposeEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long window;
    int x; int y;
    int w; int h;
    int count;
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

struct X11SelectionRequestEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long owner;
    unsigned long requestor;
    unsigned long selection;
    unsigned long target;
    unsigned long property;
    unsigned long time;
};

struct X11SelectionEvent {
    int type;
    unsigned long serial;
    int sendEvent;
    void* display;
    unsigned long requestor;
    unsigned long selection;
    unsigned long target;
    unsigned long property;
    unsigned long time;
};

// Padded to >= a real XEvent union's size (24 'long's on a 64-bit build)
// so XNextEvent has room to write whichever variant actually arrived.
struct X11EventBuf {
    long pad[24];
};

struct X11SizeHints {
    long flags;
    int x; int y; int width; int height;
    int minWidth; int minHeight; int maxWidth; int maxHeight;
    int widthInc; int heightInc;
    int minAspectX; int minAspectY; int maxAspectX; int maxAspectY;
    int baseWidth; int baseHeight;
    int winGravity;
};

// Visual's head: the channel masks tell where R, G and B go in a pixel.
struct X11VisualHead {
    void* extData;
    unsigned long visualid;
    int cls;
    unsigned long redMask;
    unsigned long greenMask;
    unsigned long blueMask;
};

struct GuiX11PollFd {
    int fd;
    short events;
    short revents;
};

#ifdef __linux__
#define GUI_X11_O_NONBLOCK 2048
#else
#define GUI_X11_O_NONBLOCK 4
#endif

struct GuiX11Window {
    unsigned long window;
    void* gc;
    void* ic;                 // input context, or null
    void* image;              // XImage over 'staging', rebuilt when the size changes
    unsigned int* staging;    // native-order copy of the buffer
    int stagingW;
    int stagingH;
};

struct GuiX11 {
    void* display;
    int screen;
    unsigned long root;
    void* visual;
    unsigned int depth;
    int rShift; int gShift; int bShift;
    void* im;
    unsigned long helper;     // hidden window: owns the clipboard selection
    int wakeR; int wakeW;     // self-pipe for wake()
    double scale;
    unsigned long aWmDelete;
    unsigned long aWmProtocols;
    unsigned long aClipboard;
    unsigned long aUtf8;
    unsigned long aTargets;
    unsigned long aText;
    unsigned long aProp;
    unsigned long aNetWmName;
    unsigned long aNetWmState;
    unsigned long aNetWmStateAbove;
    unsigned long aNetActiveWindow;
    unsigned long aMotifHints;
    unsigned long cursors[GUI_CURSOR_COUNT];
    struct String clip;
    int clipOwned;
};

static struct GuiX11 gX;
static struct GuiBackend gX11Backend;
static int gX11BackendInit = 0;

static int __gui_x11_shift(unsigned long mask) {
    if (mask == 0UL) { return 0; }
    int s = 0;
    while ((mask & 1UL) == 0UL) { mask = mask >> 1; s = s + 1; }
    return s;
}

// The scale factor for hiDpi windows: $SAFEC_GUI_SCALE, else Xft.dpi / 96
// rounded to a quarter, else 1.
static double __gui_x11_scale(void* display) {
    unsafe {
        char* env = getenv("SAFEC_GUI_SCALE");
        if (env != (char*)0) {
            double v = atof(env);
            if (v >= 0.5 && v <= 8.0) { return v; }
        }
        char* res = XResourceManagerString(display);
        if (res == (char*)0) { return 1.0; }
        unsigned long i = 0UL;
        while (res[i] != '\0') {
            // "Xft.dpi:\t192"
            if (res[i] == 'X' && res[i + 1UL] == 'f' && res[i + 2UL] == 't' && res[i + 3UL] == '.' &&
                res[i + 4UL] == 'd' && res[i + 5UL] == 'p' && res[i + 6UL] == 'i' && res[i + 7UL] == ':') {
                double dpi = atof(&res[i + 8UL]);
                if (dpi >= 48.0 && dpi <= 960.0) {
                    double s = dpi / 96.0;
                    s = (double)(int)(s * 4.0 + 0.5) / 4.0;
                    return s < 1.0 ? 1.0 : s;
                }
                return 1.0;
            }
            i = i + 1UL;
        }
        return 1.0;
    }
}

static int __gui_x11_init() {
    unsafe {
        memset((void*)&gX, 0, sizeof(struct GuiX11));
        XSetLocaleModifiers("");
        void* display = XOpenDisplay((const char*)0);
        if (display == (void*)0) { return 0; }
        gX.display = display;
        gX.screen = XDefaultScreen(display);
        gX.root = XDefaultRootWindow(display);
        gX.visual = XDefaultVisual(display, gX.screen);
        gX.depth = XDefaultDepth(display, gX.screen);
        struct X11VisualHead* vh = (struct X11VisualHead*)gX.visual;
        gX.rShift = __gui_x11_shift(vh->redMask);
        gX.gShift = __gui_x11_shift(vh->greenMask);
        gX.bShift = __gui_x11_shift(vh->blueMask);
        if (vh->redMask == 0UL) { gX.rShift = 16; gX.gShift = 8; gX.bShift = 0; }
        gX.scale = __gui_x11_scale(display);
        gX.aWmDelete = XInternAtom(display, "WM_DELETE_WINDOW", 0);
        gX.aWmProtocols = XInternAtom(display, "WM_PROTOCOLS", 0);
        gX.aClipboard = XInternAtom(display, "CLIPBOARD", 0);
        gX.aUtf8 = XInternAtom(display, "UTF8_STRING", 0);
        gX.aTargets = XInternAtom(display, "TARGETS", 0);
        gX.aText = XInternAtom(display, "TEXT", 0);
        gX.aProp = XInternAtom(display, "SAFEC_SELECTION", 0);
        gX.aNetWmName = XInternAtom(display, "_NET_WM_NAME", 0);
        gX.aNetWmState = XInternAtom(display, "_NET_WM_STATE", 0);
        gX.aNetWmStateAbove = XInternAtom(display, "_NET_WM_STATE_ABOVE", 0);
        gX.aNetActiveWindow = XInternAtom(display, "_NET_ACTIVE_WINDOW", 0);
        gX.aMotifHints = XInternAtom(display, "_MOTIF_WM_HINTS", 0);
        gX.helper = XCreateSimpleWindow(display, gX.root, 0, 0, 1U, 1U, 0U, 0UL, 0UL);
        XSelectInput(display, gX.helper, GUI_X11_PROPERTY_CHANGE_MASK);
        gX.im = XOpenIM(display, (void*)0, (char*)0, (char*)0);
        gX.clip = String::new();
        gX.clipOwned = 0;
        int fds[2];
        fds[0] = -1; fds[1] = -1;
        if (pipe(&fds[0]) == 0) {
            fcntl(fds[0], 4 /* F_SETFL */, GUI_X11_O_NONBLOCK);
            fcntl(fds[1], 4, GUI_X11_O_NONBLOCK);
            gX.wakeR = fds[0];
            gX.wakeW = fds[1];
        } else {
            gX.wakeR = -1;
            gX.wakeW = -1;
        }
        int i = 0;
        while (i < GUI_CURSOR_COUNT) { gX.cursors[i] = 0UL; i = i + 1; }
        return 1;
    }
}

// ── windows ───────────────────────────────────────────────────────────────

static struct GuiX11Window* __gui_x11_win(&GuiWindowState st) {
    unsafe { return (struct GuiX11Window*)st.native; }
}

static void __gui_x11_set_title(&GuiWindowState st, const &char title) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        int n = 0;
        while (title[n] != '\0') { n = n + 1; }
        XStoreName(gX.display, w->window, title);
        XChangeProperty(gX.display, w->window, gX.aNetWmName, gX.aUtf8, 8, 0, (const void*)title, n);
        XFlush(gX.display);
    }
}

static void __gui_x11_update_limits(&GuiWindowState st) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        struct X11SizeHints h;
        memset((void*)&h, 0, sizeof(struct X11SizeHints));
        double s = st.scale;
        if (!st.resizable) {
            h.flags = 16L | 32L;
            h.minWidth = st.pixelW; h.maxWidth = st.pixelW;
            h.minHeight = st.pixelH; h.maxHeight = st.pixelH;
        } else {
            if (st.minW > 0 || st.minH > 0) {
                h.flags = h.flags | 16L;
                h.minWidth = (int)((double)st.minW * s);
                h.minHeight = (int)((double)st.minH * s);
            }
            if (st.maxW > 0 || st.maxH > 0) {
                h.flags = h.flags | 32L;
                h.maxWidth = st.maxW > 0 ? (int)((double)st.maxW * s) : 32767;
                h.maxHeight = st.maxH > 0 ? (int)((double)st.maxH * s) : 32767;
            }
        }
        if (st.x != GUI_POS_DEFAULT && st.y != GUI_POS_DEFAULT) {
            h.flags = h.flags | 1L;   // USPosition
            h.x = st.x; h.y = st.y;
        }
        XSetWMNormalHints(gX.display, w->window, (void*)&h);
        XFlush(gX.display);
    }
}

static int __gui_x11_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        void* d = gX.display;
        st.scale = opts.hiDpi ? gX.scale : 1.0;
        st.pixelW = (int)((double)st.logicalW * st.scale + 0.5);
        st.pixelH = (int)((double)st.logicalH * st.scale + 0.5);
        int px = st.x != GUI_POS_DEFAULT ? st.x : 0;
        int py = st.y != GUI_POS_DEFAULT ? st.y : 0;
        unsigned long window = XCreateSimpleWindow(d, gX.root, px, py,
                                                   (unsigned int)st.pixelW, (unsigned int)st.pixelH,
                                                   0U, 0UL, 0UL);
        if (window == 0UL) { return 0; }
        XSelectInput(d, window,
                     GUI_X11_KEY_PRESS_MASK | GUI_X11_KEY_RELEASE_MASK |
                     GUI_X11_BUTTON_PRESS_MASK | GUI_X11_BUTTON_RELEASE_MASK |
                     GUI_X11_ENTER_WINDOW_MASK | GUI_X11_LEAVE_WINDOW_MASK |
                     GUI_X11_POINTER_MOTION_MASK | GUI_X11_EXPOSURE_MASK |
                     GUI_X11_STRUCTURE_NOTIFY_MASK | GUI_X11_FOCUS_CHANGE_MASK);
        unsigned long protocols[1];
        protocols[0] = gX.aWmDelete;
        XSetWMProtocols(d, window, (void*)&protocols[0], 1);

        struct GuiX11Window* w = (struct GuiX11Window*)alloc_zeroed_raw_(sizeof(struct GuiX11Window));
        w->window = window;
        w->gc = XCreateGC(d, window, 0UL, (void*)0);
        w->ic = (void*)0;
        if (gX.im != (void*)0) {
            w->ic = XCreateIC(gX.im, "inputStyle", 0x0008L | 0x0400L /* PreeditNothing|StatusNothing */,
                              "clientWindow", window, "focusWindow", window, (void*)0);
        }
        st.native = (void*)w;
        st.key = (unsigned long long)window;

        __gui_x11_set_title(st, (const char*)st.title.data);
        __gui_x11_update_limits(st);
        if (!opts.decorated) {
            long hints[5];
            hints[0] = 2L; hints[1] = 0L; hints[2] = 0L; hints[3] = 0L; hints[4] = 0L;
            XChangeProperty(d, window, gX.aMotifHints, gX.aMotifHints, 32, 0, (const void*)&hints[0], 5);
        }
        if (opts.topmost) {
            unsigned long above[1];
            above[0] = gX.aNetWmStateAbove;
            XChangeProperty(d, window, gX.aNetWmState, GUI_X11_XA_ATOM, 32, 0, (const void*)&above[0], 1);
        }
        if (opts.visible) { XMapWindow(d, window); }
        XFlush(d);
        if (st.x == GUI_POS_DEFAULT) { st.x = px; st.y = py; }
        return 1;
    }
}

static void __gui_x11_free_image(struct GuiX11Window* w) {
    unsafe {
        if (w->image != (void*)0) {
            // XDestroyImage would free 'data' too; the staging buffer is
            // ours, so detach it first (XImage starts with four ints, then
            // the data pointer).
            *(void**)((unsigned long)w->image + 16UL) = (void*)0;
            XDestroyImage(w->image);
            w->image = (void*)0;
        }
        if (w->staging != (unsigned int*)0) {
            dealloc(w->staging);
            w->staging = (unsigned int*)0;
        }
        w->stagingW = 0;
        w->stagingH = 0;
    }
}

static void __gui_x11_destroy(&GuiWindowState st) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        __gui_x11_free_image(w);
        if (w->ic != (void*)0) { XDestroyIC(w->ic); }
        if (w->gc != (void*)0) { XFreeGC(gX.display, w->gc); }
        XDestroyWindow(gX.display, w->window);
        XFlush(gX.display);
        dealloc(w);
        st.native = null;
    }
}

static void __gui_x11_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                              int x, int y, int rw, int rh) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        if (w->stagingW != bw || w->stagingH != bh || w->image == (void*)0) {
            __gui_x11_free_image(w);
            w->staging = (unsigned int*)alloc_zeroed_raw_((unsigned long)bw * (unsigned long)bh * 4UL);
            w->stagingW = bw;
            w->stagingH = bh;
            w->image = XCreateImage(gX.display, gX.visual, gX.depth, 2 /* ZPixmap */, 0,
                                    (const void*)w->staging, (unsigned int)bw, (unsigned int)bh, 32, 0);
            if (w->image == (void*)0) { return; }
            // Our words are in this machine's byte order; Xlib swaps for the
            // server if it differs (XImage.byte_order at offset 24).
            unsigned int probe = 1U;
            int little = (*(unsigned char*)&probe == (unsigned char)1) ? 1 : 0;
            *(int*)((unsigned long)w->image + 24UL) = little ? 0 /* LSBFirst */ : 1 /* MSBFirst */;
        }
        int rs = gX.rShift; int gs = gX.gShift; int bs = gX.bShift;
        int yy = y;
        while (yy < y + rh) {
            const unsigned char* src = (const unsigned char*)px + ((unsigned long)yy * (unsigned long)bw + (unsigned long)x) * 4UL;
            unsigned int* dst = w->staging + (unsigned long)yy * (unsigned long)bw + (unsigned long)x;
            int xx = 0;
            while (xx < rw) {
                dst[xx] = ((unsigned int)src[0] << rs) | ((unsigned int)src[1] << gs) |
                          ((unsigned int)src[2] << bs) | ((unsigned int)src[3] << 24);
                src = src + 4;
                xx = xx + 1;
            }
            yy = yy + 1;
        }
        XPutImage(gX.display, w->window, w->gc, w->image, x, y, x, y, (unsigned int)rw, (unsigned int)rh);
        XFlush(gX.display);
    }
}

static void __gui_x11_set_size(&GuiWindowState st, int lw, int lh) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        int pw = (int)((double)lw * st.scale + 0.5);
        int ph = (int)((double)lh * st.scale + 0.5);
        if (!st.resizable) {
            st.pixelW = pw; st.pixelH = ph;
            __gui_x11_update_limits(st);
        }
        XResizeWindow(gX.display, w->window, (unsigned int)pw, (unsigned int)ph);
        XFlush(gX.display);
    }
}

static void __gui_x11_set_position(&GuiWindowState st, int x, int y) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        XMoveWindow(gX.display, w->window, x, y);
        XFlush(gX.display);
    }
}

static void __gui_x11_set_visible(&GuiWindowState st, int v) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        if (v) { XMapWindow(gX.display, w->window); } else { XUnmapWindow(gX.display, w->window); }
        XFlush(gX.display);
    }
}

static void __gui_x11_focus(&GuiWindowState st) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        XRaiseWindow(gX.display, w->window);
        // Ask the window manager (EWMH), and set it directly for when there
        // is none.
        struct X11ClientMessageEvent cm;
        memset((void*)&cm, 0, sizeof(struct X11ClientMessageEvent));
        cm.type = GUI_X11_CLIENT_MESSAGE_TYPE;
        cm.window = w->window;
        cm.messageType = gX.aNetActiveWindow;
        cm.format = 32;
        cm.data0 = 1L;
        struct X11EventBuf buf;
        memset((void*)&buf, 0, sizeof(struct X11EventBuf));
        memcpy((void*)&buf, (const void*)&cm, sizeof(struct X11ClientMessageEvent));
        XSendEvent(gX.display, gX.root, 0, (1L << 20) | (1L << 19), (void*)&buf);
        XSetInputFocus(gX.display, w->window, 2 /* RevertToParent */, 0UL);
        XFlush(gX.display);
    }
}

static unsigned long __gui_x11_cursor(int c) {
    unsafe {
        if (gX.cursors[c] != 0UL) { return gX.cursors[c]; }
        unsigned long cur = 0UL;
        if (c == GUI_CURSOR_HIDDEN) {
            char bits[1];
            bits[0] = (char)0;
            unsigned long pm = XCreateBitmapFromData(gX.display, gX.root, &bits[0], 1U, 1U);
            unsigned short color[6];
            int i = 0;
            while (i < 6) { color[i] = (unsigned short)0; i = i + 1; }
            cur = XCreatePixmapCursor(gX.display, pm, pm, (void*)&color[0], (void*)&color[0], 0U, 0U);
            XFreePixmap(gX.display, pm);
        } else {
            // X cursor font glyphs.
            unsigned int shape = 68U;                           // XC_left_ptr
            if (c == GUI_CURSOR_IBEAM) { shape = 152U; }        // XC_xterm
            else if (c == GUI_CURSOR_HAND) { shape = 60U; }     // XC_hand2
            else if (c == GUI_CURSOR_CROSSHAIR) { shape = 34U; }
            else if (c == GUI_CURSOR_RESIZE_EW) { shape = 108U; }   // XC_sb_h_double_arrow
            else if (c == GUI_CURSOR_RESIZE_NS) { shape = 116U; }   // XC_sb_v_double_arrow
            else if (c == GUI_CURSOR_RESIZE_NWSE) { shape = 14U; }  // XC_bottom_right_corner
            else if (c == GUI_CURSOR_RESIZE_NESW) { shape = 12U; }  // XC_bottom_left_corner
            else if (c == GUI_CURSOR_MOVE) { shape = 52U; }         // XC_fleur
            else if (c == GUI_CURSOR_WAIT) { shape = 150U; }        // XC_watch
            else if (c == GUI_CURSOR_NOT_ALLOWED) { shape = 0U; }   // XC_X_cursor
            cur = XCreateFontCursor(gX.display, shape);
        }
        gX.cursors[c] = cur;
        return cur;
    }
}

static void __gui_x11_set_cursor(&GuiWindowState st, int c) {
    struct GuiX11Window* w = __gui_x11_win(st);
    unsafe {
        if (w == (struct GuiX11Window*)0) { return; }
        XDefineCursor(gX.display, w->window, __gui_x11_cursor(c));
        XFlush(gX.display);
    }
}

// ── keys ──────────────────────────────────────────────────────────────────

// X keysyms (layout-independent, unlike raw keycodes) to GuiKey.
static int __gui_x11_keysym(unsigned long ks) {
    if (ks == 0xFF51UL || ks == 0xFF96UL) { return GUI_KEY_LEFT; }
    if (ks == 0xFF53UL || ks == 0xFF98UL) { return GUI_KEY_RIGHT; }
    if (ks == 0xFF52UL || ks == 0xFF97UL) { return GUI_KEY_UP; }
    if (ks == 0xFF54UL || ks == 0xFF99UL) { return GUI_KEY_DOWN; }
    if (ks == 0xFF0DUL || ks == 0xFF8DUL) { return GUI_KEY_ENTER; }   // Return, KP_Enter
    if (ks == 0xFF1BUL) { return GUI_KEY_ESCAPE; }
    if (ks == 0xFF09UL || ks == 0xFE20UL) { return GUI_KEY_TAB; }      // Tab, ISO_Left_Tab
    if (ks == 0xFF08UL) { return GUI_KEY_BACKSPACE; }
    if (ks == 0x20UL)   { return GUI_KEY_SPACE; }
    if (ks == 0xFFFFUL || ks == 0xFF9FUL) { return GUI_KEY_DELETE; }
    if (ks == 0xFF50UL || ks == 0xFF95UL) { return GUI_KEY_HOME; }
    if (ks == 0xFF57UL || ks == 0xFF9CUL) { return GUI_KEY_END; }
    if (ks == 0xFF55UL || ks == 0xFF9AUL) { return GUI_KEY_PAGE_UP; }
    if (ks == 0xFF56UL || ks == 0xFF9BUL) { return GUI_KEY_PAGE_DOWN; }
    if (ks == 0xFF63UL || ks == 0xFF9EUL) { return GUI_KEY_INSERT; }
    if (ks == 0xFF67UL) { return GUI_KEY_MENU; }
    if (ks >= 0xFFBEUL && ks <= 0xFFC9UL) { return GUI_KEY_F1 + (int)(ks - 0xFFBEUL); }
    if (ks == 0xFFE1UL || ks == 0xFFE2UL) { return GUI_KEY_SHIFT; }
    if (ks == 0xFFE3UL || ks == 0xFFE4UL) { return GUI_KEY_CONTROL; }
    if (ks == 0xFFE9UL || ks == 0xFFEAUL) { return GUI_KEY_ALT; }
    if (ks == 0xFFE7UL || ks == 0xFFE8UL || ks == 0xFFEBUL || ks == 0xFFECUL) { return GUI_KEY_META; }
    if (ks >= 0x61UL && ks <= 0x7AUL) { return (int)(ks - 0x20UL); }
    if (ks >= 0x41UL && ks <= 0x5AUL) { return (int)ks; }
    if (ks >= 0x30UL && ks <= 0x39UL) { return (int)ks; }
    return GUI_KEY_UNKNOWN;
}

// The character a keysym types when no input method decoded it: Latin-1
// keysyms are their codepoint, 0x01000000+cp are Unicode.
static unsigned int __gui_x11_keysym_char(unsigned long ks) {
    if (ks >= 0x20UL && ks <= 0x7EUL) { return (unsigned int)ks; }
    if (ks >= 0xA0UL && ks <= 0xFFUL) { return (unsigned int)ks; }
    if (ks >= 0x01000100UL && ks <= 0x0110FFFFUL) { return (unsigned int)(ks - 0x01000000UL); }
    if (ks >= 0xFFAAUL && ks <= 0xFFB9UL) {   // keypad * + , - . / 0..9
        if (ks >= 0xFFB0UL) { return (unsigned int)(ks - 0xFFB0UL + 48UL); }
        if (ks == 0xFFAAUL) { return 42U; }
        if (ks == 0xFFABUL) { return 43U; }
        if (ks == 0xFFADUL) { return 45U; }
        if (ks == 0xFFAEUL) { return 46U; }
        if (ks == 0xFFAFUL) { return 47U; }
    }
    return 0U;
}

static int __gui_x11_mods(unsigned int state) {
    int m = 0;
    if ((state & 1U) != 0U) { m = m | GUI_MOD_SHIFT; }
    if ((state & 4U) != 0U) { m = m | GUI_MOD_CTRL; }
    if ((state & 8U) != 0U) { m = m | GUI_MOD_ALT; }
    if ((state & 64U) != 0U) { m = m | GUI_MOD_META; }
    return m;
}

// ── clipboard ─────────────────────────────────────────────────────────────

static void __gui_x11_serve_selection(struct X11SelectionRequestEvent* rq) {
    unsafe {
        struct X11SelectionEvent reply;
        memset((void*)&reply, 0, sizeof(struct X11SelectionEvent));
        reply.type = GUI_X11_SELECTION_NOTIFY_TYPE;
        reply.sendEvent = 1;
        reply.display = gX.display;
        reply.requestor = rq->requestor;
        reply.selection = rq->selection;
        reply.target = rq->target;
        reply.property = rq->property != 0UL ? rq->property : rq->target;
        reply.time = rq->time;
        if (!gX.clipOwned || rq->selection != gX.aClipboard) {
            reply.property = 0UL;
        } else if (rq->target == gX.aTargets) {
            unsigned long targets[4];
            targets[0] = gX.aTargets; targets[1] = gX.aUtf8; targets[2] = GUI_X11_XA_STRING; targets[3] = gX.aText;
            XChangeProperty(gX.display, rq->requestor, reply.property, GUI_X11_XA_ATOM, 32, 0,
                            (const void*)&targets[0], 4);
        } else if (rq->target == gX.aUtf8 || rq->target == gX.aText || rq->target == GUI_X11_XA_STRING) {
            unsigned long type = rq->target == GUI_X11_XA_STRING ? GUI_X11_XA_STRING : gX.aUtf8;
            XChangeProperty(gX.display, rq->requestor, reply.property, type, 8, 0,
                            (const void*)gX.clip.data, (int)gX.clip.len);
        } else {
            reply.property = 0UL;
        }
        struct X11EventBuf buf;
        memset((void*)&buf, 0, sizeof(struct X11EventBuf));
        memcpy((void*)&buf, (const void*)&reply, sizeof(struct X11SelectionEvent));
        XSendEvent(gX.display, rq->requestor, 0, 0L, (void*)&buf);
        XFlush(gX.display);
    }
}

static int __gui_x11_clip_set(const &char utf8, unsigned long n) {
    unsafe {
        gX.clip.free();
        gX.clip = String::from_n_(utf8, n);
        XSetSelectionOwner(gX.display, gX.aClipboard, gX.helper, 0UL);
        gX.clipOwned = XGetSelectionOwner(gX.display, gX.aClipboard) == gX.helper ? 1 : 0;
        XFlush(gX.display);
        return gX.clipOwned;
    }
}

static int __gui_x11_wait_fd(int timeoutMs) {
    unsafe {
        struct GuiX11PollFd fds[2];
        fds[0].fd = XConnectionNumber(gX.display);
        fds[0].events = (short)1;      // POLLIN
        fds[0].revents = (short)0;
        fds[1].fd = gX.wakeR;
        fds[1].events = (short)1;
        fds[1].revents = (short)0;
        unsigned long n = gX.wakeR >= 0 ? 2UL : 1UL;
        return poll((void*)&fds[0], n, timeoutMs);
    }
}

static int __gui_x11_clip_get(&String out) {
    out.clear();
    unsafe {
        unsigned long owner = XGetSelectionOwner(gX.display, gX.aClipboard);
        if (owner == 0UL) { return 0; }
        if (owner == gX.helper && gX.clipOwned) {
            out.push_n_((const char*)gX.clip.data, gX.clip.len);
            return 1;
        }
        unsigned long target = gX.aUtf8;
        int attempt = 0;
        while (attempt < 2) {
            XConvertSelection(gX.display, gX.aClipboard, target, gX.aProp, gX.helper, 0UL);
            XFlush(gX.display);
            double start = (double)time_mono_ns() / 1000000.0;
            struct X11EventBuf buf;
            int got = 0;
            while (!got) {
                if (XCheckTypedWindowEvent(gX.display, gX.helper, GUI_X11_SELECTION_NOTIFY_TYPE, (void*)&buf)) {
                    got = 1;
                    break;
                }
                double now = (double)time_mono_ns() / 1000000.0;
                if (now - start > 1000.0) { break; }
                __gui_x11_wait_fd(20);
            }
            if (!got) { return 0; }
            struct X11SelectionEvent* se = (struct X11SelectionEvent*)&buf;
            if (se->property == 0UL) {
                if (target == gX.aUtf8) { target = GUI_X11_XA_STRING; attempt = attempt + 1; continue; }
                return 0;
            }
            unsigned long actualType = 0UL;
            int actualFormat = 0;
            unsigned long nitems = 0UL;
            unsigned long after = 0UL;
            unsigned char* data = (unsigned char*)0;
            XGetWindowProperty(gX.display, gX.helper, gX.aProp, 0L, 0x7FFFFFFL, 1, 0UL /* AnyPropertyType */,
                               &actualType, &actualFormat, &nitems, &after, &data);
            if (data == (unsigned char*)0) { return 0; }
            if (actualFormat == 8) {
                if (actualType == GUI_X11_XA_STRING) {
                    // Latin-1 to UTF-8.
                    unsigned long i = 0UL;
                    while (i < nitems) {
                        char enc[4];
                        int k = gui_utf8_encode((unsigned int)data[i], &enc[0]);
                        out.push_n_(&enc[0], (unsigned long)k);
                        i = i + 1UL;
                    }
                } else {
                    out.push_n_((const char*)data, nitems);
                }
            }
            XFree((void*)data);
            return actualFormat == 8 ? 1 : 0;
        }
        return 0;
    }
}

// ── the event pump ────────────────────────────────────────────────────────

static void __gui_x11_post_size(&GuiWindowState st, int w, int h) {
    st.pixelW = w;
    st.pixelH = h;
    st.logicalW = (int)((double)w / st.scale + 0.5);
    st.logicalH = (int)((double)h / st.scale + 0.5);
    struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
    ev.width = w;
    ev.height = h;
    gui_backend_post(st, ev);
}

static void __gui_x11_handle(struct X11EventBuf* buf) {
    unsafe {
        struct X11AnyEvent* any = (struct X11AnyEvent*)buf;
        int t = any->type;
        if (t == GUI_X11_MAPPING_NOTIFY_TYPE) { XRefreshKeyboardMapping((void*)buf); return; }
        if (t == GUI_X11_SELECTION_REQUEST_TYPE) {
            __gui_x11_serve_selection((struct X11SelectionRequestEvent*)buf);
            return;
        }
        if (t == GUI_X11_SELECTION_CLEAR_TYPE) { gX.clipOwned = 0; return; }

        unsigned long xw = any->window;
        if (t == GUI_X11_CONFIGURE_NOTIFY_TYPE) {
            xw = ((struct X11ConfigureEvent*)buf)->window;
        }
        ?&GuiWindowState ost = gui_backend_find((unsigned long long)xw);
        if (ost.is_null()) { return; }
        struct GuiWindowState* st = (struct GuiWindowState*)ost;
        struct GuiX11Window* w = (struct GuiX11Window*)st->native;

        if (t == GUI_X11_CLIENT_MESSAGE_TYPE) {
            struct X11ClientMessageEvent* cm = (struct X11ClientMessageEvent*)buf;
            if ((unsigned long)cm->data0 == gX.aWmDelete) {
                gui_backend_post(st, GuiEvent::new(GUI_EVENT_CLOSE));
            }
            return;
        }
        if (t == GUI_X11_BUTTON_PRESS_TYPE || t == GUI_X11_BUTTON_RELEASE_TYPE) {
            struct X11ButtonEvent* be = (struct X11ButtonEvent*)buf;
            if (be->button >= 4U && be->button <= 7U) {     // wheel: 4/5 up/down, 6/7 left/right
                if (t != GUI_X11_BUTTON_PRESS_TYPE) { return; }
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCROLL);
                ev.x = (double)be->x; ev.y = (double)be->y;
                ev.scrollDx = be->button == 6U ? -1.0 : (be->button == 7U ? 1.0 : 0.0);
                ev.scrollDy = be->button == 4U ? 1.0 : (be->button == 5U ? -1.0 : 0.0);
                ev.modifiers = __gui_x11_mods(be->state);
                gui_backend_post(st, ev);
                return;
            }
            struct GuiEvent ev = GuiEvent::new(t == GUI_X11_BUTTON_PRESS_TYPE ? GUI_EVENT_MOUSE_DOWN : GUI_EVENT_MOUSE_UP);
            ev.x = (double)be->x; ev.y = (double)be->y;
            // Button1=left(1), Button2=middle, Button3=right in X11's
            // 1-based numbering.
            if (be->button == 1U) { ev.button = GUI_BUTTON_LEFT; }
            else if (be->button == 3U) { ev.button = GUI_BUTTON_RIGHT; }
            else { ev.button = GUI_BUTTON_MIDDLE; }
            ev.modifiers = __gui_x11_mods(be->state);
            gui_backend_post(st, ev);
            return;
        }
        if (t == GUI_X11_MOTION_NOTIFY_TYPE) {
            struct X11MotionEvent* me = (struct X11MotionEvent*)buf;
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOUSE_MOVE);
            ev.x = (double)me->x; ev.y = (double)me->y;
            ev.modifiers = __gui_x11_mods(me->state);
            gui_backend_post(st, ev);
            return;
        }
        if (t == GUI_X11_ENTER_NOTIFY_TYPE || t == GUI_X11_LEAVE_NOTIFY_TYPE) {
            struct X11CrossingEvent* ce = (struct X11CrossingEvent*)buf;
            struct GuiEvent ev = GuiEvent::new(t == GUI_X11_ENTER_NOTIFY_TYPE ? GUI_EVENT_MOUSE_ENTER : GUI_EVENT_MOUSE_LEAVE);
            ev.x = (double)ce->x; ev.y = (double)ce->y;
            gui_backend_post(st, ev);
            return;
        }
        if (t == GUI_X11_FOCUS_IN_TYPE || t == GUI_X11_FOCUS_OUT_TYPE) {
            struct X11FocusEvent* fe = (struct X11FocusEvent*)buf;
            if (fe->detail == 7 /* NotifyPointer */) { return; }
            int focus = t == GUI_X11_FOCUS_IN_TYPE ? 1 : 0;
            if (w != (struct GuiX11Window*)0 && w->ic != (void*)0) {
                if (focus) { XSetICFocus(w->ic); } else { XUnsetICFocus(w->ic); }
            }
            if (st->focused == focus) { return; }
            st->focused = focus;
            gui_backend_post(st, GuiEvent::new(focus ? GUI_EVENT_FOCUS_IN : GUI_EVENT_FOCUS_OUT));
            return;
        }
        if (t == GUI_X11_EXPOSE_TYPE) {
            struct X11ExposeEvent* ee = (struct X11ExposeEvent*)buf;
            if (ee->count == 0) { gui_backend_post(st, GuiEvent::new(GUI_EVENT_EXPOSE)); }
            return;
        }
        if (t == GUI_X11_KEY_PRESS_TYPE || t == GUI_X11_KEY_RELEASE_TYPE) {
            struct X11KeyEvent* ke = (struct X11KeyEvent*)buf;
            struct GuiEvent ev = GuiEvent::new(t == GUI_X11_KEY_PRESS_TYPE ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP);
            ev.modifiers = __gui_x11_mods(ke->state);
            ev.x = (double)ke->x; ev.y = (double)ke->y;
            // The keycode from the unshifted keysym (layout letters stay
            // letters), the text from the input method / shifted keysym.
            unsigned long ks0 = XLookupKeysym((void*)buf, 0);
            ev.keycode = __gui_x11_keysym(ks0);
            char text[32];
            unsigned long keysym = 0UL;
            int n = 0;
            if (t == GUI_X11_KEY_PRESS_TYPE && w != (struct GuiX11Window*)0 && w->ic != (void*)0) {
                int status = 0;
                n = Xutf8LookupString(w->ic, (void*)buf, &text[0], 31, &keysym, &status);
                if (status != 2 && status != 4) { n = 0; }   // XLookupChars / XLookupBoth carry text
                if (n < 0) { n = 0; }
            } else {
                n = XLookupString((void*)buf, &text[0], 31, &keysym, (void*)0);
                if (n > 0) {
                    unsigned int c = __gui_x11_keysym_char(keysym);
                    n = 0;
                    if (c != 0U) { n = gui_utf8_encode(c, &text[0]); }
                }
            }
            if (ev.keycode == GUI_KEY_UNKNOWN && keysym != 0UL) { ev.keycode = __gui_x11_keysym(keysym); }
            ev.codepoint = 0U;
            if (t == GUI_X11_KEY_PRESS_TYPE && n > 0 && (ev.modifiers & (GUI_MOD_CTRL | GUI_MOD_META)) == 0) {
                text[n] = '\0';
                unsigned long pos = 0UL;
                unsigned int c = gui_utf8_next_n_(&text[0], (unsigned long)n, &pos);
                if (c >= 32U && c != 127U) { ev.codepoint = c; }
                gui_backend_post(st, ev);
                // Anything after the first character (a composed sequence)
                // arrives as CHAR events.
                while (pos < (unsigned long)n) {
                    unsigned int c2 = gui_utf8_next_n_(&text[0], (unsigned long)n, &pos);
                    if (c2 >= 32U && c2 != 127U) {
                        struct GuiEvent ce = GuiEvent::new(GUI_EVENT_CHAR);
                        ce.codepoint = c2;
                        gui_backend_post(st, ce);
                    }
                }
                return;
            }
            gui_backend_post(st, ev);
            return;
        }
        if (t == GUI_X11_CONFIGURE_NOTIFY_TYPE) {
            struct X11ConfigureEvent* ce = (struct X11ConfigureEvent*)buf;
            if (ce->w != st->pixelW || ce->h != st->pixelH) {
                __gui_x11_post_size(st, ce->w, ce->h);
            }
            int rx = ce->x;
            int ry = ce->y;
            if (!ce->sendEvent) {
                // Relative to a reparenting window manager's frame: ask where
                // the client area is on the root window.
                unsigned long child = 0UL;
                XTranslateCoordinates(gX.display, ce->window, gX.root, 0, 0, &rx, &ry, &child);
            }
            int lx = (int)((double)rx / st->scale);
            int ly = (int)((double)ry / st->scale);
            if (lx != st->x || ly != st->y) {
                st->x = lx;
                st->y = ly;
                struct GuiEvent mv = GuiEvent::new(GUI_EVENT_MOVE);
                mv.x = (double)lx;
                mv.y = (double)ly;
                gui_backend_post(st, mv);
            }
            return;
        }
    }
}

static int __gui_x11_drain() {
    int any = 0;
    unsafe {
        while (XPending(gX.display) > 0) {
            struct X11EventBuf buf;
            XNextEvent(gX.display, (void*)&buf);
            if (XFilterEvent((void*)&buf, 0UL)) { continue; }   // consumed by the input method
            __gui_x11_handle(&buf);
            any = 1;
        }
    }
    return any;
}

static int __gui_x11_pump(int timeoutMs) {
    unsafe {
        XFlush(gX.display);
        int got = __gui_x11_drain();
        if (gui_backend_has_events() || timeoutMs == 0) { return got; }
        double start = (double)time_mono_ns() / 1000000.0;
        while (!gui_backend_has_events()) {
            int left = timeoutMs;
            if (timeoutMs > 0) {
                left = timeoutMs - (int)((double)time_mono_ns() / 1000000.0 - start);
                if (left <= 0) { break; }
            }
            int r = __gui_x11_wait_fd(left);
            if (gX.wakeR >= 0) {
                char drainBuf[64];
                long woke = read(gX.wakeR, (void*)&drainBuf[0], 64UL);
                if (woke > 0L) { __gui_x11_drain(); break; }
            }
            if (__gui_x11_drain()) { got = 1; }
            if (r <= 0 && timeoutMs > 0) { break; }
        }
        return got;
    }
}

static void __gui_x11_wake() {
    unsafe {
        if (gX.wakeW >= 0) {
            char b = (char)1;
            write(gX.wakeW, (const void*)&b, 1UL);
        }
    }
}

static long long __gui_x11_now_ns() { return time_mono_ns(); }

static double __gui_x11_display_scale() { return gX.scale > 0.0 ? gX.scale : 1.0; }

static struct GuiNativeHandle __gui_x11_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_X11;
    unsafe {
        h.display = (unsigned long long)(unsigned long)gX.display;
        h.visual = (unsigned long long)(unsigned long)gX.visual;
    }
    h.window = st.key;
    h.view = 0ULL;
    return h;
}

static &GuiBackend __gui_x11_backend() {
    if (!gX11BackendInit) {
        gX11Backend.name = "x11";
        gX11Backend.primaryMod = GUI_MOD_CTRL;
        gX11Backend.init = __gui_x11_init;
        gX11Backend.window_create = __gui_x11_create;
        gX11Backend.window_destroy = __gui_x11_destroy;
        gX11Backend.window_present = __gui_x11_present;
        gX11Backend.window_set_title = __gui_x11_set_title;
        gX11Backend.window_set_size = __gui_x11_set_size;
        gX11Backend.window_set_position = __gui_x11_set_position;
        gX11Backend.window_update_limits = __gui_x11_update_limits;
        gX11Backend.window_set_visible = __gui_x11_set_visible;
        gX11Backend.window_focus = __gui_x11_focus;
        gX11Backend.window_set_cursor = __gui_x11_set_cursor;
        gX11Backend.pump = __gui_x11_pump;
        gX11Backend.wake = __gui_x11_wake;
        gX11Backend.now_ns = __gui_x11_now_ns;
        gX11Backend.clipboard_set = __gui_x11_clip_set;
        gX11Backend.clipboard_get = __gui_x11_clip_get;
        gX11Backend.display_scale = __gui_x11_display_scale;
        gX11Backend.native_handle = __gui_x11_native;
        gX11BackendInit = 1;
    }
    return &gX11Backend;
}

static int __gui_x11_env_is(const char* v, const char* want) {
    unsafe {
        if (v == (const char*)0) { return 0; }
        unsigned long i = 0UL;
        while (want[i] != '\0' && v[i] == want[i]) { i = i + 1UL; }
        return (want[i] == '\0' && v[i] == '\0') ? 1 : 0;
    }
}

static ?&GuiBackend gGuiUnixChoice = null;

// Linux/BSD: $SAFEC_GUI_BACKEND (x11 | wayland | headless) if set, else
// Wayland when $WAYLAND_DISPLAY is set and libwayland-client loads and
// connects (see gui_wayland.h), else X11. Decided once.
&GuiBackend gui_platform_backend() {
    match (gGuiUnixChoice) {
        case some(b): return b;
        case null: { }
    }
    &GuiBackend choice = __gui_x11_backend();
    unsafe {
        const char* want = getenv("SAFEC_GUI_BACKEND");
        if (gui_headless_requested()) {
            choice = gui_headless_backend();
        } else if (__gui_x11_env_is(want, "x11")) {
            // as is
        } else if (__gui_x11_env_is(want, "wayland")) {
            ?&GuiBackend wl = gui_wayland_backend();
            match (wl) { case some(b): { choice = b; } case null: { } }
        } else {
            const char* wd = getenv("WAYLAND_DISPLAY");
            if (wd != (const char*)0 && wd[0] != '\0' && gui_wayland_available()) {
                ?&GuiBackend wl = gui_wayland_backend();
                match (wl) { case some(b): { choice = b; } case null: { } }
            }
        }
    }
    gGuiUnixChoice = choice;
    return choice;
}

} // namespace std
