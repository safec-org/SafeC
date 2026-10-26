// SafeC Standard Library — GUI backend: macOS (Cocoa via the Objective-C
// runtime + Core Graphics). The struct GuiBackend for AppKit (see
// gui_backend.h for the contract it implements and gui.h for the portable
// API on top).
// safeguard: link macos -framework Cocoa -lobjc
#pragma once
#include <std/stddef.h>
//
// There is no Objective-C support in safec itself — every AppKit/Core
// Graphics call here goes through the plain-C Objective-C runtime API
// (objc_msgSend et al., <objc/message.h>/<objc/runtime.h>) and Core
// Graphics' own C API, the same technique minimal Cocoa bindings in other
// languages without a real ObjC bridge use. 'objc_msgSend' is declared
// once with a generic two-pointer-argument signature and then reinterpret-
// cast (via 'typedef fn ... Name;' function-pointer types) to whatever
// concrete signature a given message send actually needs before calling
// through it — the real ABI dispatch depends entirely on the argument/
// return types at the call site, not on how the symbol itself is declared.
// Messages returning an NSRect (32 bytes) go through objc_msgSend_stret on
// x86_64, as the ABI requires there; arm64 has no _stret variant.
//
// Rendering model: a window's content view is layer-backed
// (setWantsLayer:YES); each present copies the RGBA buffer (premultiplied,
// kCGImageAlphaPremultipliedLast — exactly our format, no conversion) into
// a CGImage and sets it as the CALayer's 'contents', which AppKit
// composites to the screen on the next display pass. hiDpi windows get a
// buffer of the content size times backingScaleFactor and set the layer's
// contentsScale to match, so Retina displays show it 1:1.
//
// Events: pump() dequeues NSEvents itself (nextEventMatchingMask:...) and
// routes each to the GuiWindowState of [event window]; everything but key
// events is also passed on to [NSApp sendEvent:] so the title bar, resize
// handles and window buttons keep working (key events aren't, so AppKit
// doesn't beep about keys no responder handles). Per-window close, resize,
// move, focus and backing-scale changes come from a delegate object per
// window (class SafeCWindowDelegate, registered at first use with
// objc_allocateClassPair/class_addMethod) — windowShouldClose: answers NO
// and posts GUI_EVENT_CLOSE, so closing is the application's decision.
// Mouse enter/leave are derived from the pointer position on mouse moves.

#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_headless.h>
#include <std/gui/gui_draw.h>
#include <std/mem.h>
#include <std/time.h>
#include <std/collections/string.h>
extern void* memset(void* ptr, int val, size_t n);
extern void* memcpy(void* dst, const void* src, size_t n);

namespace std {

extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
#if defined(__x86_64__)
extern void  objc_msgSend_stret(void* recv, void* op);
#endif
extern void* objc_allocateClassPair(void* superclass, const char* name, unsigned long extraBytes);
extern void  objc_registerClassPair(void* cls);
extern int   class_addMethod(void* cls, void* sel, void* imp, const char* types);
extern void* objc_autoreleasePoolPush();
extern void  objc_autoreleasePoolPop(void* pool);

extern void* CGColorSpaceCreateDeviceRGB();
extern void* CGDataProviderCreateWithData(void* info, const void* data, unsigned long size, void* releaseFunc);
extern void* CGImageCreate(unsigned long width, unsigned long height, unsigned long bitsPerComponent,
                            unsigned long bitsPerPixel, unsigned long bytesPerRow, void* space,
                            unsigned int bitmapInfo, void* provider, const double* decode,
                            int shouldInterpolate, int intent);
extern void  CGImageRelease(void* image);
extern void  CGColorSpaceRelease(void* space);
extern void  CGDataProviderRelease(void* provider);

struct NSRect { double x; double y; double w; double h; };
struct NSPoint { double x; double y; };
struct NSSize { double w; double h; };

typedef fn void*(void*, void*) MsgSend0;
typedef fn void*(void*, void*, void*) MsgSend1;
typedef fn void*(void*, void*, unsigned long) MsgSendULong;
typedef fn void*(void*, void*, long) MsgSendLong;
typedef fn void*(void*, void*, int) MsgSendInt;
typedef fn void*(void*, void*, double) MsgSendDoubleArg;
typedef fn void*(void*, void*, struct NSRect, unsigned long, unsigned long, int) MsgSendInitRect;
typedef fn void(void*, void*, struct NSSize) MsgSendSize;
typedef fn void(void*, void*, struct NSPoint) MsgSendPointArg;
typedef fn struct NSPoint(void*, void*) MsgSendPoint;
typedef fn struct NSRect(void*, void*) MsgSendRect;
typedef fn struct NSRect(void*, void*, struct NSRect) MsgSendRectRect;
typedef fn unsigned long(void*, void*) MsgSendRetULong;
typedef fn long(void*, void*) MsgSendRetLong;
typedef fn unsigned short(void*, void*) MsgSendRetUShort;
typedef fn signed char(void*, void*) MsgSendRetBool;
typedef fn void(void*, void*, void*) MsgSendVoid1;
typedef fn void(void*, void*) MsgSendVoid0;
typedef fn void(void*, void*, void*, int) MsgSendVoid1Int;
typedef fn signed char(void*, void*, void*, void*) MsgSendBool2;
typedef fn void*(void*, void*, unsigned long, void*, void*, int) MsgSendNextEvent;
typedef fn double(void*, void*) MsgSendDouble;
typedef fn void*(void*, void*, unsigned long, struct NSPoint, unsigned long, double, long, void*,
                 short, long, long) MsgSendOtherEvent;

// ── one-time class/selector/app setup ───────────────────────────────────────

struct GuiCocoaWindow {
    void* window;       // NSWindow*
    void* view;         // its content view
    void* delegate;     // SafeCWindowDelegate instance
};

static int   gInited = 0;
static void* gNSApp;
static void* gDelegateClass;
static void* gDefaultMode;
static void* gPasteType;
static int   gCursorHidden = 0;
static struct GuiBackend gCocoaBackend;
static int gCocoaBackendInit = 0;

static void* __gui_cocoa_sel(const char* name) {
    unsafe { return sel_registerName(name); }
}

static void* __gui_cocoa_str(const char* utf8) {
    unsafe {
        MsgSend1 strMsg = (MsgSend1)objc_msgSend;
        return strMsg(objc_getClass("NSString"), __gui_cocoa_sel("stringWithUTF8String:"), (void*)utf8);
    }
}

static struct NSRect __gui_cocoa_rect(void* obj, const char* sel) {
    unsafe {
#if defined(__x86_64__)
        MsgSendRect m = (MsgSendRect)objc_msgSend_stret;
#else
        MsgSendRect m = (MsgSendRect)objc_msgSend;
#endif
        return m(obj, __gui_cocoa_sel(sel));
    }
}

static double __gui_cocoa_screen_height() {
    unsafe {
        MsgSend0 m = (MsgSend0)objc_msgSend;
        void* screen = m(objc_getClass("NSScreen"), __gui_cocoa_sel("mainScreen"));
        if (screen == (void*)0) { return 0.0; }
        struct NSRect r = __gui_cocoa_rect(screen, "frame");
        return r.h;
    }
}

static ?&GuiWindowState __gui_cocoa_find(void* nsWindow) {
    unsafe { return gui_backend_find((unsigned long long)(unsigned long)nsWindow); }
}

static void* __gui_cocoa_notif_window(void* notif) {
    unsafe {
        MsgSend0 m = (MsgSend0)objc_msgSend;
        return m(notif, __gui_cocoa_sel("object"));
    }
}

// The content area's size in points and its top-left on screen (y down).
static void __gui_cocoa_update_geometry(struct GuiWindowState* st, int postEvents) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st->native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        struct NSRect cv = __gui_cocoa_rect(w->view, "frame");
        struct NSRect fr = __gui_cocoa_rect(w->window, "frame");
#if defined(__x86_64__)
        MsgSendRectRect toContent = (MsgSendRectRect)objc_msgSend_stret;
#else
        MsgSendRectRect toContent = (MsgSendRectRect)objc_msgSend;
#endif
        struct NSRect content = toContent(w->window, __gui_cocoa_sel("contentRectForFrameRect:"), fr);
        int lw = (int)(cv.w + 0.5);
        int lh = (int)(cv.h + 0.5);
        int px = (int)((double)lw * st->scale + 0.5);
        int py = (int)((double)lh * st->scale + 0.5);
        if (lw > 0 && lh > 0 && (px != st->pixelW || py != st->pixelH || lw != st->logicalW || lh != st->logicalH)) {
            st->logicalW = lw;
            st->logicalH = lh;
            st->pixelW = px;
            st->pixelH = py;
            if (postEvents) {
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
                ev.width = px;
                ev.height = py;
                gui_backend_post(st, ev);
            }
        }
        double sh = __gui_cocoa_screen_height();
        int nx = (int)content.x;
        int ny = (int)(sh - (content.y + content.h));
        if (nx != st->x || ny != st->y) {
            st->x = nx;
            st->y = ny;
            if (postEvents) {
                struct GuiEvent mv = GuiEvent::new(GUI_EVENT_MOVE);
                mv.x = (double)nx;
                mv.y = (double)ny;
                gui_backend_post(st, mv);
            }
        }
    }
}

signed char __gui_cocoa_window_should_close(void* recv, void* cmd, void* sender) {
    ?&GuiWindowState ost = __gui_cocoa_find(sender);
    unsafe {
        if (!ost.is_null()) { gui_backend_post((struct GuiWindowState*)ost, GuiEvent::new(GUI_EVENT_CLOSE)); }
    }
    return (signed char)0;   // NO: the application closes it (GuiWindow::destroy)
}

void __gui_cocoa_window_did_resize(void* recv, void* cmd, void* notif) {
    ?&GuiWindowState ost = __gui_cocoa_find(__gui_cocoa_notif_window(notif));
    unsafe { if (!ost.is_null()) { __gui_cocoa_update_geometry((struct GuiWindowState*)ost, 1); } }
}

void __gui_cocoa_window_did_move(void* recv, void* cmd, void* notif) {
    ?&GuiWindowState ost = __gui_cocoa_find(__gui_cocoa_notif_window(notif));
    unsafe { if (!ost.is_null()) { __gui_cocoa_update_geometry((struct GuiWindowState*)ost, 1); } }
}

void __gui_cocoa_window_did_become_key(void* recv, void* cmd, void* notif) {
    ?&GuiWindowState ost = __gui_cocoa_find(__gui_cocoa_notif_window(notif));
    unsafe {
        if (ost.is_null()) { return; }
        struct GuiWindowState* st = (struct GuiWindowState*)ost;
        if (!st->focused) {
            st->focused = 1;
            gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_IN));
        }
    }
}

void __gui_cocoa_window_did_resign_key(void* recv, void* cmd, void* notif) {
    ?&GuiWindowState ost = __gui_cocoa_find(__gui_cocoa_notif_window(notif));
    unsafe {
        if (ost.is_null()) { return; }
        struct GuiWindowState* st = (struct GuiWindowState*)ost;
        if (st->focused) {
            st->focused = 0;
            gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_OUT));
        }
    }
}

void __gui_cocoa_window_did_change_backing(void* recv, void* cmd, void* notif) {
    ?&GuiWindowState ost = __gui_cocoa_find(__gui_cocoa_notif_window(notif));
    unsafe {
        if (ost.is_null()) { return; }
        struct GuiWindowState* st = (struct GuiWindowState*)ost;
        if (!st->hiDpi) { return; }
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st->native;
        MsgSendDouble scaleMsg = (MsgSendDouble)objc_msgSend;
        double s = scaleMsg(w->window, __gui_cocoa_sel("backingScaleFactor"));
        if (s <= 0.0 || s == st->scale) { return; }
        st->scale = s;
        st->pixelW = (int)((double)st->logicalW * s + 0.5);
        st->pixelH = (int)((double)st->logicalH * s + 0.5);
        struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCALE);
        ev.width = st->pixelW;
        ev.height = st->pixelH;
        gui_backend_post(st, ev);
    }
}

static int __gui_cocoa_init() {
    if (gInited) { return 1; }
    unsafe {
        void* appCls = objc_getClass("NSApplication");
        if (appCls == (void*)0) { return 0; }
        MsgSend0 sharedApp = (MsgSend0)objc_msgSend;
        gNSApp = sharedApp(appCls, __gui_cocoa_sel("sharedApplication"));

        // NSApplicationActivationPolicyRegular = 0
        MsgSendLong setPolicy = (MsgSendLong)objc_msgSend;
        setPolicy(gNSApp, __gui_cocoa_sel("setActivationPolicy:"), 0L);

        MsgSendVoid0 finishLaunching = (MsgSendVoid0)objc_msgSend;
        finishLaunching(gNSApp, __gui_cocoa_sel("finishLaunching"));

        MsgSendInt activate = (MsgSendInt)objc_msgSend;
        activate(gNSApp, __gui_cocoa_sel("activateIgnoringOtherApps:"), 1);

        void* nsObjectCls = objc_getClass("NSObject");
        gDelegateClass = objc_allocateClassPair(nsObjectCls, "SafeCWindowDelegate", 0UL);
        if (gDelegateClass != (void*)0) {
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowShouldClose:"),
                            (void*)__gui_cocoa_window_should_close, "c@:@");
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowDidResize:"),
                            (void*)__gui_cocoa_window_did_resize, "v@:@");
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowDidMove:"),
                            (void*)__gui_cocoa_window_did_move, "v@:@");
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowDidBecomeKey:"),
                            (void*)__gui_cocoa_window_did_become_key, "v@:@");
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowDidResignKey:"),
                            (void*)__gui_cocoa_window_did_resign_key, "v@:@");
            class_addMethod(gDelegateClass, __gui_cocoa_sel("windowDidChangeBackingProperties:"),
                            (void*)__gui_cocoa_window_did_change_backing, "v@:@");
            objc_registerClassPair(gDelegateClass);
        } else {
            gDelegateClass = objc_getClass("SafeCWindowDelegate");   // registered earlier in this process
        }
        // [NSString stringWithUTF8String:] returns autoreleased objects;
        // keep these two alive for the process.
        MsgSend0 retainMsg = (MsgSend0)objc_msgSend;
        gDefaultMode = retainMsg(__gui_cocoa_str("kCFRunLoopDefaultMode"), __gui_cocoa_sel("retain"));
        gPasteType = retainMsg(__gui_cocoa_str("public.utf8-plain-text"), __gui_cocoa_sel("retain"));
    }
    gInited = 1;
    return 1;
}

// ── window creation ──────────────────────────────────────────────────────────

static void __gui_cocoa_apply_limits(&GuiWindowState st) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        MsgSendSize setSize = (MsgSendSize)objc_msgSend;
        struct NSSize mn;
        mn.w = (double)(st.minW > 0 ? st.minW : 1);
        mn.h = (double)(st.minH > 0 ? st.minH : 1);
        setSize(w->window, __gui_cocoa_sel("setContentMinSize:"), mn);
        struct NSSize mx;
        mx.w = st.maxW > 0 ? (double)st.maxW : 100000.0;
        mx.h = st.maxH > 0 ? (double)st.maxH : 100000.0;
        setSize(w->window, __gui_cocoa_sel("setContentMaxSize:"), mx);
    }
}

static int __gui_cocoa_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        void* winCls = objc_getClass("NSWindow");
        MsgSend0 allocMsg = (MsgSend0)objc_msgSend;
        void* winAlloc = allocMsg(winCls, __gui_cocoa_sel("alloc"));

        double sh = __gui_cocoa_screen_height();
        struct NSRect rect;
        rect.w = (double)st.logicalW;
        rect.h = (double)st.logicalH;
        rect.x = st.x != GUI_POS_DEFAULT ? (double)st.x : 100.0;
        rect.y = st.y != GUI_POS_DEFAULT ? sh - (double)st.y - rect.h : 100.0;
        // styleMask: titled(1) | closable(2) | miniaturizable(4) | resizable(8); borderless 0
        unsigned long style = 0UL;
        if (opts.decorated) {
            style = 1UL | 2UL | 4UL;
            if (opts.resizable) { style = style | 8UL; }
        }
        MsgSendInitRect initMsg = (MsgSendInitRect)objc_msgSend;
        void* nsWin = initMsg(winAlloc, __gui_cocoa_sel("initWithContentRect:styleMask:backing:defer:"),
                              rect, style, 2UL, 0);
        if (nsWin == (void*)0) { objc_autoreleasePoolPop(pool); return 0; }

        // We own the window (GuiWindow::destroy releases it); AppKit must
        // not release it again when it closes.
        MsgSendInt setReleased = (MsgSendInt)objc_msgSend;
        setReleased(nsWin, __gui_cocoa_sel("setReleasedWhenClosed:"), 0);

        MsgSendVoid1 setTitle = (MsgSendVoid1)objc_msgSend;
        setTitle(nsWin, __gui_cocoa_sel("setTitle:"), __gui_cocoa_str((const char*)st.title.data));

        if (st.x == GUI_POS_DEFAULT || st.y == GUI_POS_DEFAULT) {
            MsgSendVoid0 centerMsg = (MsgSendVoid0)objc_msgSend;
            centerMsg(nsWin, __gui_cocoa_sel("center"));
        }
        MsgSendInt acceptMoves = (MsgSendInt)objc_msgSend;
        acceptMoves(nsWin, __gui_cocoa_sel("setAcceptsMouseMovedEvents:"), 1);
        if (opts.topmost) {
            MsgSendLong setLevel = (MsgSendLong)objc_msgSend;
            setLevel(nsWin, __gui_cocoa_sel("setLevel:"), 3L);          // NSFloatingWindowLevel
        }

        // Delegate (close/resize/move/focus/scale).
        MsgSend0 delAlloc = (MsgSend0)objc_msgSend;
        void* delInstance = delAlloc(gDelegateClass, __gui_cocoa_sel("alloc"));
        MsgSend0 delInit = (MsgSend0)objc_msgSend;
        delInstance = delInit(delInstance, __gui_cocoa_sel("init"));
        MsgSendVoid1 setDelegate = (MsgSendVoid1)objc_msgSend;
        setDelegate(nsWin, __gui_cocoa_sel("setDelegate:"), delInstance);

        // Layer-backed content view, for CGImage-based presentation.
        MsgSend0 contentView = (MsgSend0)objc_msgSend;
        void* view = contentView(nsWin, __gui_cocoa_sel("contentView"));
        MsgSendInt setWantsLayer = (MsgSendInt)objc_msgSend;
        setWantsLayer(view, __gui_cocoa_sel("setWantsLayer:"), 1);

        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)alloc_zeroed_raw_(sizeof(struct GuiCocoaWindow));
        w->window = nsWin;
        w->view = view;
        w->delegate = delInstance;
        st.native = (void*)w;
        st.key = (unsigned long long)(unsigned long)nsWin;

        st.scale = 1.0;
        if (opts.hiDpi) {
            MsgSendDouble scaleMsg = (MsgSendDouble)objc_msgSend;
            double s = scaleMsg(nsWin, __gui_cocoa_sel("backingScaleFactor"));
            if (s > 0.0) { st.scale = s; }
        }
        st.pixelW = (int)((double)st.logicalW * st.scale + 0.5);
        st.pixelH = (int)((double)st.logicalH * st.scale + 0.5);
        MsgSend0 layerMsg = (MsgSend0)objc_msgSend;
        void* layer = layerMsg(view, __gui_cocoa_sel("layer"));
        if (layer != (void*)0) {
            MsgSendDoubleArg setScale = (MsgSendDoubleArg)objc_msgSend;
            setScale(layer, __gui_cocoa_sel("setContentsScale:"), st.scale);
        }
        __gui_cocoa_apply_limits(st);
        if (!opts.resizable) {
            struct NSSize fixed;
            fixed.w = (double)st.logicalW;
            fixed.h = (double)st.logicalH;
            MsgSendSize setSize = (MsgSendSize)objc_msgSend;
            setSize(nsWin, __gui_cocoa_sel("setContentMinSize:"), fixed);
            setSize(nsWin, __gui_cocoa_sel("setContentMaxSize:"), fixed);
        }
        __gui_cocoa_update_geometry((struct GuiWindowState*)st, 0);

        if (opts.visible) {
            MsgSendVoid1 orderFront = (MsgSendVoid1)objc_msgSend;
            orderFront(nsWin, __gui_cocoa_sel("makeKeyAndOrderFront:"), (void*)0);
        }
        objc_autoreleasePoolPop(pool);
        return 1;
    }
}

static void __gui_cocoa_destroy(&GuiWindowState st) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        MsgSendVoid1 setDelegate = (MsgSendVoid1)objc_msgSend;
        setDelegate(w->window, __gui_cocoa_sel("setDelegate:"), (void*)0);
        MsgSendVoid0 closeMsg = (MsgSendVoid0)objc_msgSend;
        closeMsg(w->window, __gui_cocoa_sel("close"));
        MsgSendVoid0 releaseMsg = (MsgSendVoid0)objc_msgSend;
        releaseMsg(w->window, __gui_cocoa_sel("release"));
        releaseMsg(w->delegate, __gui_cocoa_sel("release"));
        dealloc_raw_((void*)w);
        st.native = null;
    }
}

// ── presentation ─────────────────────────────────────────────────────────────

// CGDataProviderReleaseDataCallback: frees the copy made for one present.
void __gui_cocoa_release_pixels(void* info, const void* data, unsigned long size) {
    unsafe { dealloc_raw_(info); }
}

static void __gui_cocoa_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                                int x, int y, int rw, int rh) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        void* pool = objc_autoreleasePoolPush();
        MsgSend0 layerMsg = (MsgSend0)objc_msgSend;
        void* layer = layerMsg(w->view, __gui_cocoa_sel("layer"));
        if (layer == (void*)0) { objc_autoreleasePoolPop(pool); return; }

        // The layer keeps reading its contents after this returns, and the
        // window's buffer is reallocated on resize: hand it a copy, freed
        // by the data provider's release callback.
        unsigned long rowBytes = (unsigned long)bw * 4UL;
        unsigned long total = rowBytes * (unsigned long)bh;
        void* copy = alloc_raw_(total);
        if (copy == (void*)0) { objc_autoreleasePoolPop(pool); return; }
        memcpy(copy, (const void*)px, total);
        void* space = CGColorSpaceCreateDeviceRGB();
        void* provider = CGDataProviderCreateWithData(copy, (const void*)copy, total,
                                                      (void*)__gui_cocoa_release_pixels);
        // bitmapInfo = kCGImageAlphaPremultipliedLast (1)
        void* image = CGImageCreate((unsigned long)bw, (unsigned long)bh, 8UL, 32UL,
                                    rowBytes, space, 1U, provider, (const double*)0, 0, 0);

        MsgSendVoid1 setContents = (MsgSendVoid1)objc_msgSend;
        setContents(layer, __gui_cocoa_sel("setContents:"), image);

        CGImageRelease(image);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(space);
        objc_autoreleasePoolPop(pool);
    }
}

// ── window management ───────────────────────────────────────────────────────

static void __gui_cocoa_set_title(&GuiWindowState st, const &char title) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        void* pool = objc_autoreleasePoolPush();
        MsgSendVoid1 setTitle = (MsgSendVoid1)objc_msgSend;
        setTitle(w->window, __gui_cocoa_sel("setTitle:"), __gui_cocoa_str(title));
        objc_autoreleasePoolPop(pool);
    }
}

static void __gui_cocoa_set_size(&GuiWindowState st, int lw, int lh) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        struct NSSize s;
        s.w = (double)lw;
        s.h = (double)lh;
        MsgSendSize setSize = (MsgSendSize)objc_msgSend;
        setSize(w->window, __gui_cocoa_sel("setContentSize:"), s);   // windowDidResize: reports it
    }
}

static void __gui_cocoa_set_position(&GuiWindowState st, int x, int y) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        // Places the frame's top-left corner (the title bar's, for a titled
        // window): AppKit coordinates have y up from the bottom of the main
        // screen.
        struct NSPoint p;
        p.x = (double)x;
        p.y = __gui_cocoa_screen_height() - (double)y;
        MsgSendPointArg setTop = (MsgSendPointArg)objc_msgSend;
        setTop(w->window, __gui_cocoa_sel("setFrameTopLeftPoint:"), p);
    }
}

static void __gui_cocoa_set_visible(&GuiWindowState st, int v) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        MsgSendVoid1 m = (MsgSendVoid1)objc_msgSend;
        m(w->window, __gui_cocoa_sel(v ? "orderFront:" : "orderOut:"), (void*)0);
    }
}

static void __gui_cocoa_focus(&GuiWindowState st) {
    unsafe {
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w == (struct GuiCocoaWindow*)0) { return; }
        MsgSendInt activate = (MsgSendInt)objc_msgSend;
        activate(gNSApp, __gui_cocoa_sel("activateIgnoringOtherApps:"), 1);
        MsgSendVoid1 orderFront = (MsgSendVoid1)objc_msgSend;
        orderFront(w->window, __gui_cocoa_sel("makeKeyAndOrderFront:"), (void*)0);
    }
}

static void __gui_cocoa_apply_cursor(int c) {
    unsafe {
        MsgSend0 m = (MsgSend0)objc_msgSend;
        void* cls = objc_getClass("NSCursor");
        if (c == GUI_CURSOR_HIDDEN) {
            if (!gCursorHidden) {
                MsgSendVoid0 hide = (MsgSendVoid0)objc_msgSend;
                hide(cls, __gui_cocoa_sel("hide"));
                gCursorHidden = 1;
            }
            return;
        }
        if (gCursorHidden) {
            MsgSendVoid0 unhide = (MsgSendVoid0)objc_msgSend;
            unhide(cls, __gui_cocoa_sel("unhide"));
            gCursorHidden = 0;
        }
        const char* name = "arrowCursor";
        if (c == GUI_CURSOR_IBEAM) { name = "IBeamCursor"; }
        else if (c == GUI_CURSOR_HAND) { name = "pointingHandCursor"; }
        else if (c == GUI_CURSOR_CROSSHAIR) { name = "crosshairCursor"; }
        else if (c == GUI_CURSOR_RESIZE_EW) { name = "resizeLeftRightCursor"; }
        else if (c == GUI_CURSOR_RESIZE_NS) { name = "resizeUpDownCursor"; }
        else if (c == GUI_CURSOR_MOVE) { name = "openHandCursor"; }
        else if (c == GUI_CURSOR_NOT_ALLOWED) { name = "operationNotAllowedCursor"; }
        void* cur = m(cls, __gui_cocoa_sel(name));
        if (cur != (void*)0) {
            MsgSendVoid0 setMsg = (MsgSendVoid0)objc_msgSend;
            setMsg(cur, __gui_cocoa_sel("set"));
        }
    }
}

static void __gui_cocoa_set_cursor(&GuiWindowState st, int c) {
    if (st.hovered) { __gui_cocoa_apply_cursor(c); }
}

// ── event loop ────────────────────────────────────────────────────────────────

static int __gui_cocoa_keycode(unsigned short native) {
    int n = (int)native;
    if (n == 123) { return GUI_KEY_LEFT; }
    if (n == 124) { return GUI_KEY_RIGHT; }
    if (n == 126) { return GUI_KEY_UP; }
    if (n == 125) { return GUI_KEY_DOWN; }
    if (n == 36 || n == 76) { return GUI_KEY_ENTER; }
    if (n == 53) { return GUI_KEY_ESCAPE; }
    if (n == 48) { return GUI_KEY_TAB; }
    if (n == 51) { return GUI_KEY_BACKSPACE; }
    if (n == 49) { return GUI_KEY_SPACE; }
    if (n == 117) { return GUI_KEY_DELETE; }
    if (n == 115) { return GUI_KEY_HOME; }
    if (n == 119) { return GUI_KEY_END; }
    if (n == 116) { return GUI_KEY_PAGE_UP; }
    if (n == 121) { return GUI_KEY_PAGE_DOWN; }
    if (n == 114) { return GUI_KEY_INSERT; }
    if (n == 56 || n == 60) { return GUI_KEY_SHIFT; }
    if (n == 59 || n == 62) { return GUI_KEY_CONTROL; }
    if (n == 58 || n == 61) { return GUI_KEY_ALT; }
    if (n == 55 || n == 54) { return GUI_KEY_META; }
    if (n == 110) { return GUI_KEY_MENU; }
    // F1..F12
    if (n == 122) { return GUI_KEY_F1; }
    if (n == 120) { return GUI_KEY_F1 + 1; }
    if (n == 99)  { return GUI_KEY_F1 + 2; }
    if (n == 118) { return GUI_KEY_F1 + 3; }
    if (n == 96)  { return GUI_KEY_F1 + 4; }
    if (n == 97)  { return GUI_KEY_F1 + 5; }
    if (n == 98)  { return GUI_KEY_F1 + 6; }
    if (n == 100) { return GUI_KEY_F1 + 7; }
    if (n == 101) { return GUI_KEY_F1 + 8; }
    if (n == 109) { return GUI_KEY_F1 + 9; }
    if (n == 103) { return GUI_KEY_F1 + 10; }
    if (n == 111) { return GUI_KEY_F1 + 11; }
    // ANSI letter and digit keys (kVK_ANSI_*): position-based like X11's
    // unshifted keysym and Win32's virtual key.
    if (n >= 0 && n <= 50) {
        const char* map = "ASDFHGZXCV?BQWERYT123465=97-80]OU[IP?LJ'K;\\,/NM.";
        char c;
        unsafe { c = map[n]; }
        if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) { return (int)c; }
    }
    return GUI_KEY_UNKNOWN;
}

static int __gui_cocoa_mods(unsigned long flags) {
    int m = 0;
    if ((flags & (1UL << 17)) != 0UL) { m = m | GUI_MOD_SHIFT; }
    if ((flags & (1UL << 18)) != 0UL) { m = m | GUI_MOD_CTRL; }
    if ((flags & (1UL << 19)) != 0UL) { m = m | GUI_MOD_ALT; }
    if ((flags & (1UL << 20)) != 0UL) { m = m | GUI_MOD_META; }
    return m;
}

// Routes one NSEvent; returns 1 if it should also go to [NSApp sendEvent:].
static int __gui_cocoa_route(void* ev) {
    unsafe {
        MsgSendRetULong typeMsg = (MsgSendRetULong)objc_msgSend;
        unsigned long etype = typeMsg(ev, __gui_cocoa_sel("type"));
        int isKey = (etype == 10UL || etype == 11UL || etype == 12UL) ? 1 : 0;
        if (etype == 15UL) { return 0; }                 // our own wake() event
        MsgSend0 winMsg = (MsgSend0)objc_msgSend;
        void* nsWin = winMsg(ev, __gui_cocoa_sel("window"));
        if (nsWin == (void*)0 && isKey) { nsWin = winMsg(gNSApp, __gui_cocoa_sel("keyWindow")); }
        if (nsWin == (void*)0) { return isKey ? 0 : 1; }
        ?&GuiWindowState ost = __gui_cocoa_find(nsWin);
        if (ost.is_null()) { return isKey ? 0 : 1; }
        struct GuiWindowState* st = (struct GuiWindowState*)ost;

        MsgSendRetULong flagsMsg = (MsgSendRetULong)objc_msgSend;
        int mods = __gui_cocoa_mods(flagsMsg(ev, __gui_cocoa_sel("modifierFlags")));

        int isMouse = (etype >= 1UL && etype <= 7UL) || etype == 22UL || etype == 25UL || etype == 26UL || etype == 27UL;
        double ex = 0.0;
        double ey = 0.0;
        if (isMouse) {
            MsgSendPoint locMsg = (MsgSendPoint)objc_msgSend;
            struct NSPoint loc = locMsg(ev, __gui_cocoa_sel("locationInWindow"));
            // Window coordinates have y up from the content's bottom edge.
            ex = loc.x * st->scale;
            ey = ((double)st->logicalH - loc.y) * st->scale;
        }

        if (etype == 1UL || etype == 3UL || etype == 25UL || etype == 2UL || etype == 4UL || etype == 26UL) {
            int down = (etype == 1UL || etype == 3UL || etype == 25UL) ? 1 : 0;
            struct GuiEvent e = GuiEvent::new(down ? GUI_EVENT_MOUSE_DOWN : GUI_EVENT_MOUSE_UP);
            e.x = ex; e.y = ey;
            e.button = (etype == 1UL || etype == 2UL) ? GUI_BUTTON_LEFT
                     : ((etype == 3UL || etype == 4UL) ? GUI_BUTTON_RIGHT : GUI_BUTTON_MIDDLE);
            // Control-click is the one-button right click.
            if (e.button == GUI_BUTTON_LEFT && (mods & GUI_MOD_CTRL) != 0) { e.button = GUI_BUTTON_RIGHT; }
            e.modifiers = mods;
            gui_backend_post(st, e);
            return 1;
        }
        if (etype == 5UL || etype == 6UL || etype == 7UL || etype == 27UL) {
            int inside = (ex >= 0.0 && ey >= 0.0 && ex < (double)st->pixelW && ey < (double)st->pixelH) ? 1 : 0;
            if (inside && !st->hovered) {
                struct GuiEvent en = GuiEvent::new(GUI_EVENT_MOUSE_ENTER);
                en.x = ex; en.y = ey;
                gui_backend_post(st, en);
                __gui_cocoa_apply_cursor(st->cursor);
            } else if (!inside && st->hovered && etype == 5UL) {
                struct GuiEvent lv = GuiEvent::new(GUI_EVENT_MOUSE_LEAVE);
                lv.x = ex; lv.y = ey;
                gui_backend_post(st, lv);
                __gui_cocoa_apply_cursor(GUI_CURSOR_ARROW);
            }
            struct GuiEvent e = GuiEvent::new(GUI_EVENT_MOUSE_MOVE);
            e.x = ex; e.y = ey;
            e.modifiers = mods;
            gui_backend_post(st, e);
            return 1;
        }
        if (etype == 22UL) {
            MsgSendDouble dxMsg = (MsgSendDouble)objc_msgSend;
            double dx = dxMsg(ev, __gui_cocoa_sel("scrollingDeltaX"));
            double dy = dxMsg(ev, __gui_cocoa_sel("scrollingDeltaY"));
            MsgSendRetBool preciseMsg = (MsgSendRetBool)objc_msgSend;
            if (preciseMsg(ev, __gui_cocoa_sel("hasPreciseScrollingDeltas")) != (signed char)0) {
                dx = dx / 10.0;   // trackpad points to "lines"
                dy = dy / 10.0;
            }
            struct GuiEvent e = GuiEvent::new(GUI_EVENT_SCROLL);
            e.x = ex; e.y = ey;
            e.scrollDx = dx;
            e.scrollDy = dy;
            e.modifiers = mods;
            gui_backend_post(st, e);
            return 1;
        }
        if (etype == 10UL || etype == 11UL) {
            MsgSendRetUShort kcMsg = (MsgSendRetUShort)objc_msgSend;
            unsigned short nativeKc = kcMsg(ev, __gui_cocoa_sel("keyCode"));
            struct GuiEvent e = GuiEvent::new(etype == 10UL ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP);
            e.keycode = __gui_cocoa_keycode(nativeKc);
            e.modifiers = mods;
            if (etype == 10UL && (mods & (GUI_MOD_META | GUI_MOD_CTRL)) == 0) {
                // The typed text: [event characters] (an NSString) as UTF-8.
                MsgSend0 charsMsg = (MsgSend0)objc_msgSend;
                void* nsChars = charsMsg(ev, __gui_cocoa_sel("characters"));
                if (nsChars != (void*)0) {
                    MsgSend0 utf8Msg = (MsgSend0)objc_msgSend;
                    const char* s = (const char*)utf8Msg(nsChars, __gui_cocoa_sel("UTF8String"));
                    if (s != (const char*)0) {
                        unsigned long n = 0UL;
                        while (s[n] != '\0') { n = n + 1UL; }
                        unsigned long pos = 0UL;
                        unsigned int c = gui_utf8_next_n_(s, n, &pos);
                        // Function keys come as private-use characters.
                        if (c >= 32U && c != 127U && !(c >= 0xF700U && c <= 0xF8FFU)) { e.codepoint = c; }
                        gui_backend_post(st, e);
                        while (pos < n) {
                            unsigned int c2 = gui_utf8_next_n_(s, n, &pos);
                            if (c2 >= 32U && c2 != 127U && !(c2 >= 0xF700U && c2 <= 0xF8FFU)) {
                                struct GuiEvent ce = GuiEvent::new(GUI_EVENT_CHAR);
                                ce.codepoint = c2;
                                gui_backend_post(st, ce);
                            }
                        }
                        return 0;
                    }
                }
            }
            gui_backend_post(st, e);
            return 0;
        }
        return isKey ? 0 : 1;
    }
}

static int __gui_cocoa_pump(int timeoutMs) {
    int any = 0;
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        MsgSend0 dateMsg = (MsgSend0)objc_msgSend;
        void* dateCls = objc_getClass("NSDate");
        MsgSendNextEvent nextEvent = (MsgSendNextEvent)objc_msgSend;
        MsgSendVoid1 sendEvent = (MsgSendVoid1)objc_msgSend;
        int first = 1;
        while (1) {
            void* until;
            if (first && timeoutMs != 0 && !gui_backend_has_events()) {
                if (timeoutMs < 0) {
                    until = dateMsg(dateCls, __gui_cocoa_sel("distantFuture"));
                } else {
                    MsgSendDoubleArg sinceNow = (MsgSendDoubleArg)objc_msgSend;
                    until = sinceNow(dateCls, __gui_cocoa_sel("dateWithTimeIntervalSinceNow:"),
                                     (double)timeoutMs / 1000.0);
                }
            } else {
                until = dateMsg(dateCls, __gui_cocoa_sel("distantPast"));
            }
            first = 0;
            void* ev = nextEvent(gNSApp, __gui_cocoa_sel("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                                 0xFFFFFFFFFFFFFFFFUL, until, gDefaultMode, 1);
            if (ev == (void*)0) { break; }
            if (__gui_cocoa_route(ev)) { sendEvent(gNSApp, __gui_cocoa_sel("sendEvent:"), ev); }
            any = 1;
        }
        MsgSendVoid0 updateWindows = (MsgSendVoid0)objc_msgSend;
        updateWindows(gNSApp, __gui_cocoa_sel("updateWindows"));
        objc_autoreleasePoolPop(pool);
    }
    return any;
}

static void __gui_cocoa_wake() {
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        struct NSPoint p;
        p.x = 0.0;
        p.y = 0.0;
        MsgSendOtherEvent other = (MsgSendOtherEvent)objc_msgSend;
        void* ev = other(objc_getClass("NSEvent"),
                         __gui_cocoa_sel("otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:"),
                         15UL /* NSEventTypeApplicationDefined */, p, 0UL, 0.0, 0L, (void*)0, (short)0, 0L, 0L);
        if (ev != (void*)0) {
            MsgSendVoid1Int post = (MsgSendVoid1Int)objc_msgSend;
            post(gNSApp, __gui_cocoa_sel("postEvent:atStart:"), ev, 0);
        }
        objc_autoreleasePoolPop(pool);
    }
}

static long long __gui_cocoa_now_ns() { return time_mono_ns(); }

static int __gui_cocoa_clip_set(const &char utf8, unsigned long n) {
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        MsgSend0 m = (MsgSend0)objc_msgSend;
        void* pb = m(objc_getClass("NSPasteboard"), __gui_cocoa_sel("generalPasteboard"));
        if (pb == (void*)0) { objc_autoreleasePoolPop(pool); return 0; }
        MsgSendRetLong clearMsg = (MsgSendRetLong)objc_msgSend;
        clearMsg(pb, __gui_cocoa_sel("clearContents"));
        // NUL-terminated copy for stringWithUTF8String:.
        char* z = (char*)alloc_raw_(n + 1UL);
        memcpy((void*)z, (const void*)utf8, n);
        z[n] = '\0';
        void* str = __gui_cocoa_str(z);
        dealloc_raw_((void*)z);
        MsgSendBool2 setStr = (MsgSendBool2)objc_msgSend;
        signed char ok = setStr(pb, __gui_cocoa_sel("setString:forType:"), str, gPasteType);
        objc_autoreleasePoolPop(pool);
        return ok != (signed char)0 ? 1 : 0;
    }
}

static int __gui_cocoa_clip_get(&String out) {
    out.clear();
    unsafe {
        void* pool = objc_autoreleasePoolPush();
        MsgSend0 m = (MsgSend0)objc_msgSend;
        void* pb = m(objc_getClass("NSPasteboard"), __gui_cocoa_sel("generalPasteboard"));
        int ok = 0;
        if (pb != (void*)0) {
            MsgSend1 get = (MsgSend1)objc_msgSend;
            void* str = get(pb, __gui_cocoa_sel("stringForType:"), gPasteType);
            if (str != (void*)0) {
                const char* s = (const char*)m(str, __gui_cocoa_sel("UTF8String"));
                if (s != (const char*)0) {
                    out.push(s);
                    ok = 1;
                }
            }
        }
        objc_autoreleasePoolPop(pool);
        return ok;
    }
}

static double __gui_cocoa_display_scale() {
    unsafe {
        MsgSend0 m = (MsgSend0)objc_msgSend;
        void* screen = m(objc_getClass("NSScreen"), __gui_cocoa_sel("mainScreen"));
        if (screen == (void*)0) { return 1.0; }
        MsgSendDouble scaleMsg = (MsgSendDouble)objc_msgSend;
        double s = scaleMsg(screen, __gui_cocoa_sel("backingScaleFactor"));
        return s > 0.0 ? s : 1.0;
    }
}

static struct GuiNativeHandle __gui_cocoa_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_COCOA;
    h.display = 0ULL;
    h.window = st.key;
    h.view = 0ULL;
    h.visual = 0ULL;
    unsafe {
        h.display = (unsigned long long)(unsigned long)gNSApp;
        struct GuiCocoaWindow* w = (struct GuiCocoaWindow*)st.native;
        if (w != (struct GuiCocoaWindow*)0) { h.view = (unsigned long long)(unsigned long)w->view; }
    }
    return h;
}

static &GuiBackend __gui_native_backend() {
    if (!gCocoaBackendInit) {
        gCocoaBackend.name = "cocoa";
        gCocoaBackend.primaryMod = GUI_MOD_META;
        gCocoaBackend.init = __gui_cocoa_init;
        gCocoaBackend.window_create = __gui_cocoa_create;
        gCocoaBackend.window_destroy = __gui_cocoa_destroy;
        gCocoaBackend.window_present = __gui_cocoa_present;
        gCocoaBackend.window_set_title = __gui_cocoa_set_title;
        gCocoaBackend.window_set_size = __gui_cocoa_set_size;
        gCocoaBackend.window_set_position = __gui_cocoa_set_position;
        gCocoaBackend.window_update_limits = __gui_cocoa_apply_limits;
        gCocoaBackend.window_set_visible = __gui_cocoa_set_visible;
        gCocoaBackend.window_focus = __gui_cocoa_focus;
        gCocoaBackend.window_set_cursor = __gui_cocoa_set_cursor;
        gCocoaBackend.pump = __gui_cocoa_pump;
        gCocoaBackend.wake = __gui_cocoa_wake;
        gCocoaBackend.now_ns = __gui_cocoa_now_ns;
        gCocoaBackend.clipboard_set = __gui_cocoa_clip_set;
        gCocoaBackend.clipboard_get = __gui_cocoa_clip_get;
        gCocoaBackend.display_scale = __gui_cocoa_display_scale;
        gCocoaBackend.native_handle = __gui_cocoa_native;
        gCocoaBackendInit = 1;
    }
    return &gCocoaBackend;
}

// ── system-font text (Cocoa-only) ───────────────────────────────────────────
// Real anti-aliased system-font text, via Core Graphics'
// CGContextShowTextAtPoint (a plain-C, no-Objective-C-message-send text
// API, unlike the rest of AppKit — deprecated since 10.9 in favor of Core
// Text, but still present and functional). Kept for compatibility: the
// portable way to draw TrueType text on every backend is gui_canvas.h's
// GuiCanvas::fill_text() with a gui_text.h GuiTypeface (e.g. loaded from
// /System/Library/Fonts). Call this only from code that already branches
// on '#ifdef __APPLE__'. It stays a free function rather than a GuiWindow
// method: only this backend defines it.
extern void* CGBitmapContextCreate(void* data, unsigned long width, unsigned long height,
                                    unsigned long bitsPerComponent, unsigned long bytesPerRow,
                                    void* colorSpace, unsigned int bitmapInfo);
extern void  CGContextRelease(void* ctx);
extern void  CGContextSetRGBFillColor(void* ctx, double r, double g, double b, double a);
extern void  CGContextSelectFont(void* ctx, const char* fontName, double size, unsigned int textEncoding);
extern void  CGContextSetTextDrawingMode(void* ctx, unsigned int mode);
extern void  CGContextShowTextAtPoint(void* ctx, double x, double y, const char* text, unsigned long length);

int gui_draw_text_system(&GuiWindow win, int x, int y, const &char text,
                          struct GuiColor color, double fontSize, const &char fontName) {
    unsafe {
        void* space = CGColorSpaceCreateDeviceRGB();
        unsigned long rowBytes = (unsigned long)win.width * 4UL;
        // Draws directly into win.pixels (no separate blit) — same
        // buffer GuiWindow::present() reads every frame.
        void* ctx = CGBitmapContextCreate((void*)win.pixels, (unsigned long)win.width,
                                           (unsigned long)win.height, 8UL, rowBytes, space, 1U);
        if (ctx == (void*)0) { CGColorSpaceRelease(space); return 0; }

        CGContextSetRGBFillColor(ctx, (double)color.r / 255.0, (double)color.g / 255.0,
                                  (double)color.b / 255.0, (double)color.a / 255.0);
        CGContextSelectFont(ctx, fontName, fontSize, 1U /* kCGEncodingMacRoman */);
        CGContextSetTextDrawingMode(ctx, 0U /* kCGTextFill */);

        unsigned long textLen = 0UL;
        while (text[textLen] != '\0') { textLen = textLen + 1UL; }
        // CGBitmapContextCreate's coordinate origin is bottom-left; flip
        // Y (and nudge up by the font size so 'y' means "top of the text"
        // like GuiWindow::draw_text(), not the Quartz baseline).
        double cgY = (double)win.height - (double)y - fontSize;
        CGContextShowTextAtPoint(ctx, (double)x, cgY, text, textLen);

        CGContextRelease(ctx);
        CGColorSpaceRelease(space);
        return 1;
    }
}

// $SAFEC_GUI_BACKEND=headless selects the in-memory backend (gui_headless.h).
&GuiBackend gui_platform_backend() {
    if (gui_headless_requested()) { return gui_headless_backend(); }
    return __gui_native_backend();
}

} // namespace std
