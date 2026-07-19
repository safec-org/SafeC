// SafeC Standard Library — GUI backend: macOS (Cocoa via the Objective-C
// runtime + Core Graphics). See gui.h for the portable API this implements.
#pragma once
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
//
// Rendering model: a window is layer-backed (NSView.wantsLayer = YES);
// each gui_present() wraps the app's raw RGBA8888 buffer in a CGImage
// (CGDataProviderCreateWithData over the existing buffer — no extra copy)
// and sets it as the CALayer's 'contents', which AppKit composites to the
// screen on the next display pass. No Metal/OpenGL context, no GPU
// drawing calls — this is a software framebuffer, matching gui.h's
// documented scope.
//
// Window-close detection: a tiny custom NSObject subclass
// ('SafeCWindowDelegate') is registered at first use via
// objc_allocateClassPair/class_addMethod, implementing 'windowWillClose:'
// to set a process-wide flag. This only disambiguates "some window
// closed," not which one — correct and sufficient for the common
// single-window case gui.h targets; a real multi-window close-per-window
// story would need an ivar or associated-object lookup back to the
// specific GuiWindow, left as a documented limitation for now.

#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/mem.sc>

namespace std {

extern void* objc_getClass(const char* name);
extern void* sel_registerName(const char* name);
extern void* objc_msgSend(void* recv, void* op);
extern void* objc_allocateClassPair(void* superclass, const char* name, unsigned long extraBytes);
extern void  objc_registerClassPair(void* cls);
extern int   class_addMethod(void* cls, void* sel, void* imp, const char* types);

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

typedef fn void*(void*, void*) MsgSend0;
typedef fn void*(void*, void*, void*) MsgSend1;
typedef fn void*(void*, void*, unsigned long) MsgSendULong;
typedef fn void*(void*, void*, long) MsgSendLong;
typedef fn void*(void*, void*, int) MsgSendInt;
typedef fn void*(void*, void*, struct NSRect, unsigned long, unsigned long, int) MsgSendInitRect;
typedef fn struct NSPoint(void*, void*) MsgSendPoint;
typedef fn struct NSPoint(void*, void*, void*) MsgSendPoint1;
typedef fn unsigned long(void*, void*) MsgSendRetULong;
typedef fn unsigned short(void*, void*) MsgSendRetUShort;
typedef fn void(void*, void*, void*) MsgSendVoid1;
typedef fn void(void*, void*) MsgSendVoid0;
typedef fn void*(void*, void*, unsigned long, void*, void*, int) MsgSendNextEvent;
typedef fn double(void*, void*) MsgSendDouble;

// ── one-time class/selector/app setup ───────────────────────────────────────

static int   gInited = 0;
static void* gNSApp;
static void* gDelegateClass;
static int   gCloseRequested = 0; // see file header: single-window granularity

void __gui_cocoa_window_will_close(void* recv, void* cmd, void* notif) {
    gCloseRequested = 1;
}

void __gui_cocoa_init() {
    if (gInited) { return; }
    unsafe {
        void* appCls = objc_getClass("NSApplication");
        MsgSend0 sharedApp = (MsgSend0)objc_msgSend;
        gNSApp = sharedApp(appCls, sel_registerName("sharedApplication"));

        // NSApplicationActivationPolicyRegular = 0
        MsgSendLong setPolicy = (MsgSendLong)objc_msgSend;
        setPolicy(gNSApp, sel_registerName("setActivationPolicy:"), 0L);

        MsgSendInt activate = (MsgSendInt)objc_msgSend;
        activate(gNSApp, sel_registerName("activateIgnoringOtherApps:"), 1);

        MsgSendVoid0 finishLaunching = (MsgSendVoid0)objc_msgSend;
        finishLaunching(gNSApp, sel_registerName("finishLaunching"));

        void* nsObjectCls = objc_getClass("NSObject");
        gDelegateClass = objc_allocateClassPair(nsObjectCls, "SafeCWindowDelegate", 0UL);
        class_addMethod(gDelegateClass, sel_registerName("windowWillClose:"),
                         (void*)__gui_cocoa_window_will_close, "v@:@");
        objc_registerClassPair(gDelegateClass);
    }
    gInited = 1;
}

// ── window creation ──────────────────────────────────────────────────────────

struct GuiWindow gui_create_window(const char* title, int width, int height) {
    __gui_cocoa_init();

    struct GuiWindow win;
    win.platform = (void*)0;
    win.width = width;
    win.height = height;
    win.shouldClose = 0;

    unsigned long npix = (unsigned long)width * (unsigned long)height * 4UL;
    unsafe {
        win.pixels = (&heap unsigned char)alloc(npix);
        memset((void*)win.pixels, 0, npix);

        void* winCls = objc_getClass("NSWindow");
        MsgSend0 allocMsg = (MsgSend0)objc_msgSend;
        void* winAlloc = allocMsg(winCls, sel_registerName("alloc"));

        struct NSRect rect;
        rect.x = 100.0; rect.y = 100.0; rect.w = (double)width; rect.h = (double)height;
        // styleMask: titled(1) | closable(2) | miniaturizable(4) | resizable(8)
        MsgSendInitRect initMsg = (MsgSendInitRect)objc_msgSend;
        void* nsWin = initMsg(winAlloc, sel_registerName("initWithContentRect:styleMask:backing:defer:"),
                               rect, 15UL, 2UL, 0);

        void* strCls = objc_getClass("NSString");
        MsgSend1 strMsg = (MsgSend1)objc_msgSend;
        void* nsTitle = strMsg(strCls, sel_registerName("stringWithUTF8String:"), (void*)title);
        MsgSendVoid1 setTitle = (MsgSendVoid1)objc_msgSend;
        setTitle(nsWin, sel_registerName("setTitle:"), nsTitle);

        MsgSendVoid0 centerMsg = (MsgSendVoid0)objc_msgSend;
        centerMsg(nsWin, sel_registerName("center"));

        // Delegate (close detection).
        MsgSend0 delAlloc = (MsgSend0)objc_msgSend;
        void* delInstance = delAlloc(gDelegateClass, sel_registerName("alloc"));
        MsgSend0 delInit = (MsgSend0)objc_msgSend;
        delInstance = delInit(delInstance, sel_registerName("init"));
        MsgSendVoid1 setDelegate = (MsgSendVoid1)objc_msgSend;
        setDelegate(nsWin, sel_registerName("setDelegate:"), delInstance);

        // Layer-backed content view, for CGImage-based presentation.
        MsgSend0 contentView = (MsgSend0)objc_msgSend;
        void* view = contentView(nsWin, sel_registerName("contentView"));
        MsgSendInt setWantsLayer = (MsgSendInt)objc_msgSend;
        setWantsLayer(view, sel_registerName("setWantsLayer:"), 1);

        MsgSendInt orderFront = (MsgSendInt)objc_msgSend;
        orderFront(nsWin, sel_registerName("makeKeyAndOrderFront:"), 0);

        win.platform = nsWin;
    }
    return win;
}

// ── presentation ─────────────────────────────────────────────────────────────

void gui_present(struct GuiWindow* win) {
    int noPlatform = 0;
    unsafe { if (win->platform == (void*)0) { noPlatform = 1; } }
    if (noPlatform) { return; }
    unsafe {
        MsgSend0 contentView = (MsgSend0)objc_msgSend;
        void* view = contentView(win->platform, sel_registerName("contentView"));
        MsgSend0 layerMsg = (MsgSend0)objc_msgSend;
        void* layer = layerMsg(view, sel_registerName("layer"));

        void* space = CGColorSpaceCreateDeviceRGB();
        unsigned long rowBytes = (unsigned long)win->width * 4UL;
        unsigned long total = rowBytes * (unsigned long)win->height;
        void* provider = CGDataProviderCreateWithData((void*)0, (const void*)win->pixels, total, (void*)0);
        // bitmapInfo = kCGImageAlphaPremultipliedLast (1)
        void* image = CGImageCreate((unsigned long)win->width, (unsigned long)win->height, 8UL, 32UL,
                                     rowBytes, space, 1U, provider, (const double*)0, 0, 0);

        MsgSendVoid1 setContents = (MsgSendVoid1)objc_msgSend;
        setContents(layer, sel_registerName("setContents:"), image);

        CGImageRelease(image);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(space);
    }
}

void gui_set_pixel(struct GuiWindow* win, int x, int y, unsigned int rgba) {
    int oob = 0;
    unsafe { if (x < 0 || y < 0 || x >= win->width || y >= win->height) { oob = 1; } }
    if (oob) { return; }
    unsafe {
        unsigned long idx = ((unsigned long)y * (unsigned long)win->width + (unsigned long)x) * 4UL;
        win->pixels[idx + 0UL] = (unsigned char)((rgba >> 24) & 0xFFU);
        win->pixels[idx + 1UL] = (unsigned char)((rgba >> 16) & 0xFFU);
        win->pixels[idx + 2UL] = (unsigned char)((rgba >> 8)  & 0xFFU);
        win->pixels[idx + 3UL] = (unsigned char)(rgba & 0xFFU);
    }
}

// ── event loop ────────────────────────────────────────────────────────────────

static int __gui_cocoa_keycode(unsigned short native) {
    if (native == 123U) { return GUI_KEY_LEFT; }
    if (native == 124U) { return GUI_KEY_RIGHT; }
    if (native == 126U) { return GUI_KEY_UP; }
    if (native == 125U) { return GUI_KEY_DOWN; }
    if (native == 36U)  { return GUI_KEY_ENTER; }
    if (native == 53U)  { return GUI_KEY_ESCAPE; }
    if (native == 48U)  { return GUI_KEY_TAB; }
    if (native == 51U)  { return GUI_KEY_BACKSPACE; }
    if (native == 49U)  { return GUI_KEY_SPACE; }
    if (native == 117U) { return GUI_KEY_DELETE; }
    return GUI_KEY_UNKNOWN;
}

int gui_poll_event(struct GuiWindow* win, struct GuiEvent* outEvent) {
    unsafe { outEvent->kind = GUI_EVENT_NONE; outEvent->defaultPrevented = 0; }

    if (gCloseRequested) {
        gCloseRequested = 0;
        unsafe {
            win->shouldClose = 1;
            outEvent->kind = GUI_EVENT_CLOSE;
        }
        return 1;
    }

    int handled = 0;
    unsafe {
        void* dateCls = objc_getClass("NSDate");
        MsgSend0 pastDateMsg = (MsgSend0)objc_msgSend;
        void* pastDate = pastDateMsg(dateCls, sel_registerName("distantPast"));

        void* strCls = objc_getClass("NSString");
        MsgSend1 strMsg = (MsgSend1)objc_msgSend;
        void* mode = strMsg(strCls, sel_registerName("stringWithUTF8String:"), (void*)"kCFRunLoopDefaultMode");

        MsgSendNextEvent nextEvent = (MsgSendNextEvent)objc_msgSend;
        void* ev = nextEvent(gNSApp, sel_registerName("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                              0xFFFFFFFFFFFFFFFFUL, pastDate, mode, 1);
        if (ev != (void*)0) {
            MsgSendVoid1 sendEvent = (MsgSendVoid1)objc_msgSend;
            sendEvent(gNSApp, sel_registerName("sendEvent:"), ev);

            MsgSendRetULong typeMsg = (MsgSendRetULong)objc_msgSend;
            unsigned long etype = typeMsg(ev, sel_registerName("type"));

            MsgSendPoint locMsg = (MsgSendPoint)objc_msgSend;
            struct NSPoint loc = locMsg(ev, sel_registerName("locationInWindow"));
            outEvent->x = loc.x;
            outEvent->y = (double)win->height - loc.y; // flip to top-left origin

            if (etype == 1UL) { outEvent->kind = GUI_EVENT_MOUSE_DOWN; outEvent->button = 0; handled = 1; }
            else if (etype == 2UL) { outEvent->kind = GUI_EVENT_MOUSE_UP; outEvent->button = 0; handled = 1; }
            else if (etype == 3UL) { outEvent->kind = GUI_EVENT_MOUSE_DOWN; outEvent->button = 1; handled = 1; }
            else if (etype == 4UL) { outEvent->kind = GUI_EVENT_MOUSE_UP; outEvent->button = 1; handled = 1; }
            else if (etype == 5UL || etype == 6UL || etype == 7UL) { outEvent->kind = GUI_EVENT_MOUSE_MOVE; handled = 1; }
            else if (etype == 22UL) {
                MsgSendDouble dxMsg = (MsgSendDouble)objc_msgSend;
                double dx = dxMsg(ev, sel_registerName("scrollingDeltaX"));
                MsgSendDouble dyMsg = (MsgSendDouble)objc_msgSend;
                double dy = dyMsg(ev, sel_registerName("scrollingDeltaY"));
                outEvent->kind = GUI_EVENT_SCROLL;
                outEvent->scrollDx = dx;
                outEvent->scrollDy = dy;
                handled = 1;
            }
            else if (etype == 10UL || etype == 11UL) {
                MsgSendRetUShort kcMsg = (MsgSendRetUShort)objc_msgSend;
                unsigned short nativeKc = kcMsg(ev, sel_registerName("keyCode"));
                outEvent->kind = (etype == 10UL) ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP;
                outEvent->keycode = __gui_cocoa_keycode(nativeKc);
                outEvent->codepoint = 0U;
                if (etype == 10UL) {
                    // Also decode the typed character (if any) into
                    // codepoint, so a focused text widget can insert it
                    // without a separate CHAR event — [event characters]
                    // returns an NSString; UTF8String's first byte is
                    // enough for the printable-ASCII range gui_draw's
                    // built-in font covers (0x20-0x7E).
                    MsgSend0 charsMsg = (MsgSend0)objc_msgSend;
                    void* nsChars = charsMsg(ev, sel_registerName("characters"));
                    if (nsChars != (void*)0) {
                        MsgSend0 utf8Msg = (MsgSend0)objc_msgSend;
                        void* cstr = utf8Msg(nsChars, sel_registerName("UTF8String"));
                        if (cstr != (void*)0) {
                            const char* s = (const char*)cstr;
                            if (s[0] != '\0') {
                                outEvent->codepoint = (unsigned int)(unsigned char)s[0];
                            }
                        }
                    }
                }
                handled = 1;
            }
        }
    }
    return handled;
}

void gui_destroy_window(struct GuiWindow* win) {
    unsafe {
        if (win->platform != (void*)0) {
            MsgSendVoid0 closeMsg = (MsgSendVoid0)objc_msgSend;
            closeMsg(win->platform, sel_registerName("close"));
        }
        if ((void*)win->pixels != (void*)0) {
            dealloc((void*)win->pixels);
        }
        win->platform = (void*)0;
    }
}

// ── system-font text (Cocoa-only) ───────────────────────────────────────────
// gui_draw.h's gui_draw_text() always uses the one built-in 8x8 bitmap
// font, portable across all four backends. This is the platform-specific
// alternative for Cocoa specifically: real anti-aliased system-font text,
// via Core Graphics' CGContextShowTextAtPoint (a plain-C, no-Objective-C-
// message-send text API, unlike the rest of AppKit — deprecated since
// 10.9 in favor of Core Text, but still present and functional; chosen
// over Core Text here because it avoids a large amount of additional
// CoreFoundation/CoreText ceremony — CFAttributedStringCreate,
// CTLineCreateWithAttributedString, dictionary-based attribute keys — for
// what this needs: draw one plain string in one color at one size).
// There is no equivalent for gui_win32.sc/gui_x11.sc/gui_fb.sc — real
// system-font rendering there needs GDI (Win32) or Xft/fontconfig (X11),
// and bare metal has no OS font service to call into at all (see
// gui_font.h's SCXF format for that case: convert any open-licensed font
// you like to SCXF and gui_font_load() it — the "open fonts for bare
// metal" story is that loader, not a font this library bundles). Call
// this only from code that already branches on '#ifdef __APPLE__'.
extern void* CGBitmapContextCreate(void* data, unsigned long width, unsigned long height,
                                    unsigned long bitsPerComponent, unsigned long bytesPerRow,
                                    void* colorSpace, unsigned int bitmapInfo);
extern void  CGContextRelease(void* ctx);
extern void  CGContextSetRGBFillColor(void* ctx, double r, double g, double b, double a);
extern void  CGContextSelectFont(void* ctx, const char* fontName, double size, unsigned int textEncoding);
extern void  CGContextSetTextDrawingMode(void* ctx, unsigned int mode);
extern void  CGContextShowTextAtPoint(void* ctx, double x, double y, const char* text, unsigned long length);

int gui_draw_text_system(struct GuiWindow* win, int x, int y, const char* text,
                          struct GuiColor color, double fontSize, const char* fontName) {
    unsafe {
        void* space = CGColorSpaceCreateDeviceRGB();
        unsigned long rowBytes = (unsigned long)win->width * 4UL;
        // Draws directly into win->pixels (no separate blit) — same
        // buffer gui_present() reads every frame.
        void* ctx = CGBitmapContextCreate((void*)win->pixels, (unsigned long)win->width,
                                           (unsigned long)win->height, 8UL, rowBytes, space, 1U);
        if (ctx == (void*)0) { CGColorSpaceRelease(space); return 0; }

        CGContextSetRGBFillColor(ctx, (double)color.r / 255.0, (double)color.g / 255.0,
                                  (double)color.b / 255.0, (double)color.a / 255.0);
        CGContextSelectFont(ctx, fontName, fontSize, 1U /* kCGEncodingMacRoman */);
        CGContextSetTextDrawingMode(ctx, 0U /* kCGTextFill */);

        unsigned long textLen = 0UL;
        while (text[textLen] != '\0') { textLen = textLen + 1UL; }
        // CGBitmapContextCreate's coordinate origin is bottom-left; flip
        // Y (and nudge up by the font size so 'y' means "top of the text"
        // like gui_draw_text(), not the Quartz baseline).
        double cgY = (double)win->height - (double)y - fontSize;
        CGContextShowTextAtPoint(ctx, (double)x, cgY, text, textLen);

        CGContextRelease(ctx);
        CGColorSpaceRelease(space);
        return 1;
    }
}

} // namespace std
