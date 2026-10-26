// SafeC Standard Library — GUI backend: Wayland (see gui_wayland.h for
// what it covers and gui_backend.h for the contract it implements).
//
// libwayland-client is dlopen()ed, never linked, and driven through its
// stable C ABI: wl_proxy_marshal_array_flags() (or, before libwayland 1.20,
// wl_proxy_marshal_array_constructor_versioned() + wl_proxy_destroy()) to
// send requests with their arguments in a union wl_argument array, and
// wl_proxy_add_dispatcher() to receive every event of an object in one
// function with (opcode, arguments) — so no per-event C listener structs
// are needed. The core interfaces (wl_surface_interface, ...) are data
// symbols of the library; the extension protocols' struct wl_interface
// tables (xdg-shell, xdg-decoration, viewporter, fractional-scale,
// cursor-shape) are built here at run time from their XML's request/event
// signatures — what wayland-scanner's C output would contain.
// safeguard: link linux -ldl
// safeguard: deb-recommends linux libwayland-client0 libxkbcommon0 libwayland-cursor0
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_wayland.h>

#if defined(__linux__)
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
extern int close(int fd);
extern int poll(void* fds, unsigned long nfds, int timeout);
extern int fcntl(int fd, int cmd, ...);
extern int open(const char* path, int flags, ...);
extern int mkstemp(char* templ);
extern int unlink(const char* path);
extern int ftruncate(int fd, long length);
extern void* mmap(void* addr, unsigned long len, int prot, int flags, int fd, long off);
extern int munmap(void* addr, unsigned long len);
extern void* dlopen(const char* file, int mode);
extern void* dlsym(void* handle, const char* name);
#endif

namespace std {

#if defined(__linux__)

// ── libwayland-client, libxkbcommon, libwayland-cursor (dlopen) ──────────

typedef fn void*(const char*) WlConnect_;
typedef fn void(void*) WlVoidP_;
typedef fn int(void*) WlIntP_;
typedef fn void*(void*, unsigned int, const void*, unsigned int, unsigned int, void*) WlMarshalFlags_;
typedef fn void*(void*, unsigned int, void*, const void*, unsigned int) WlMarshalCtorV_;
typedef fn int(void*, void*, const void*, void*) WlAddDispatcher_;
typedef fn unsigned int(void*) WlGetVersion_;
typedef fn void*(void*) WlGetUserData_;

typedef fn void*(int) XkbContextNew_;
typedef fn void*(void*, const char*, int, int) XkbKeymapFromString_;
typedef fn void(void*) XkbUnref_;
typedef fn void*(void*) XkbStateNew_;
typedef fn int(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) XkbUpdateMask_;
typedef fn unsigned int(void*, unsigned int) XkbKeyU_;
typedef fn int(void*, const char*, int) XkbModActive_;
typedef fn int(void*, unsigned int, unsigned int, unsigned int, const unsigned int**) XkbSymsByLevel_;
typedef fn int(void*, unsigned int) XkbKeyRepeats_;
typedef fn void*(void*, const char*, int) XkbComposeTableNew_;
typedef fn void*(void*, int) XkbComposeStateNew_;
typedef fn int(void*, unsigned int) XkbComposeFeed_;
typedef fn int(void*) XkbComposeStatus_;
typedef fn unsigned int(void*) XkbComposeKeysym_;
typedef fn void(void*) XkbComposeReset_;

typedef fn void*(const char*, int, void*) WlCursorThemeLoad_;
typedef fn void(void*) WlCursorThemeDestroy_;
typedef fn void*(void*, const char*) WlCursorThemeGet_;
typedef fn void*(void*) WlCursorImageBuffer_;

struct GuiWlLib {
    int loaded;                        // 0 not tried, 1 ok, -1 unavailable
    WlConnect_ display_connect;
    WlVoidP_ display_disconnect;
    WlIntP_ display_flush;
    WlIntP_ display_get_fd;
    WlIntP_ display_dispatch_pending;
    WlIntP_ display_roundtrip;
    WlIntP_ display_prepare_read;
    WlIntP_ display_read_events;
    WlVoidP_ display_cancel_read;
    WlIntP_ display_get_error;
    WlMarshalFlags_ marshal_flags;     // null before libwayland 1.20
    WlMarshalCtorV_ marshal_ctor;
    WlVoidP_ proxy_destroy;
    WlAddDispatcher_ add_dispatcher;
    WlGetVersion_ get_version;
    WlGetUserData_ get_user_data;
    // core interfaces (data symbols)
    void* iRegistry; void* iCallback; void* iCompositor; void* iSurface; void* iRegion;
    void* iShm; void* iShmPool; void* iBuffer; void* iSeat; void* iPointer; void* iKeyboard;
    void* iOutput; void* iDataDeviceManager; void* iDataDevice; void* iDataSource; void* iDataOffer;

    int xkb;                           // libxkbcommon loaded
    XkbContextNew_ xkb_context_new;
    XkbKeymapFromString_ xkb_keymap_new_from_string;
    XkbUnref_ xkb_keymap_unref;
    XkbUnref_ xkb_state_unref;
    XkbStateNew_ xkb_state_new;
    XkbUpdateMask_ xkb_state_update_mask;
    XkbKeyU_ xkb_state_key_get_one_sym;
    XkbKeyU_ xkb_state_key_get_utf32;
    XkbKeyU_ xkb_state_key_get_layout;
    XkbModActive_ xkb_state_mod_name_is_active;
    XkbSymsByLevel_ xkb_keymap_key_get_syms_by_level;
    XkbKeyRepeats_ xkb_keymap_key_repeats;
    XkbComposeTableNew_ xkb_compose_table_new_from_locale;   // may be null
    XkbComposeStateNew_ xkb_compose_state_new;
    XkbComposeFeed_ xkb_compose_state_feed;
    XkbComposeStatus_ xkb_compose_state_get_status;
    XkbComposeKeysym_ xkb_compose_state_get_one_sym;
    XkbComposeReset_ xkb_compose_state_reset;

    int cursor;                        // libwayland-cursor loaded
    WlCursorThemeLoad_ cursor_theme_load;
    WlCursorThemeDestroy_ cursor_theme_destroy;
    WlCursorThemeGet_ cursor_theme_get_cursor;
    WlCursorImageBuffer_ cursor_image_get_buffer;
};

static struct GuiWlLib gWlLib;

static void* __wl_sym(void* lib, const char* name, &int ok) {
    unsafe {
        void* p = dlsym(lib, name);
        if (p == (void*)0) { *ok = 0; }
        return p;
    }
}

static void* __wl_open2(const char* a, const char* b) {
    unsafe {
        void* h = dlopen(a, 2 /* RTLD_NOW */);
        if (h == (void*)0 && b != (const char*)0) { h = dlopen(b, 2); }
        return h;
    }
}

static int __wl_load() {
    if (gWlLib.loaded != 0) { return gWlLib.loaded > 0 ? 1 : 0; }
    gWlLib.loaded = -1;
    unsafe {
        void* lib = __wl_open2("libwayland-client.so.0", "libwayland-client.so");
        if (lib == (void*)0) { return 0; }
        int ok = 1;
        gWlLib.display_connect = (WlConnect_)__wl_sym(lib, "wl_display_connect", &ok);
        gWlLib.display_disconnect = (WlVoidP_)__wl_sym(lib, "wl_display_disconnect", &ok);
        gWlLib.display_flush = (WlIntP_)__wl_sym(lib, "wl_display_flush", &ok);
        gWlLib.display_get_fd = (WlIntP_)__wl_sym(lib, "wl_display_get_fd", &ok);
        gWlLib.display_dispatch_pending = (WlIntP_)__wl_sym(lib, "wl_display_dispatch_pending", &ok);
        gWlLib.display_roundtrip = (WlIntP_)__wl_sym(lib, "wl_display_roundtrip", &ok);
        gWlLib.display_prepare_read = (WlIntP_)__wl_sym(lib, "wl_display_prepare_read", &ok);
        gWlLib.display_read_events = (WlIntP_)__wl_sym(lib, "wl_display_read_events", &ok);
        gWlLib.display_cancel_read = (WlVoidP_)__wl_sym(lib, "wl_display_cancel_read", &ok);
        gWlLib.display_get_error = (WlIntP_)__wl_sym(lib, "wl_display_get_error", &ok);
        gWlLib.marshal_ctor = (WlMarshalCtorV_)__wl_sym(lib, "wl_proxy_marshal_array_constructor_versioned", &ok);
        gWlLib.proxy_destroy = (WlVoidP_)__wl_sym(lib, "wl_proxy_destroy", &ok);
        gWlLib.add_dispatcher = (WlAddDispatcher_)__wl_sym(lib, "wl_proxy_add_dispatcher", &ok);
        gWlLib.get_version = (WlGetVersion_)__wl_sym(lib, "wl_proxy_get_version", &ok);
        gWlLib.get_user_data = (WlGetUserData_)__wl_sym(lib, "wl_proxy_get_user_data", &ok);
        gWlLib.iRegistry = __wl_sym(lib, "wl_registry_interface", &ok);
        gWlLib.iCallback = __wl_sym(lib, "wl_callback_interface", &ok);
        gWlLib.iCompositor = __wl_sym(lib, "wl_compositor_interface", &ok);
        gWlLib.iSurface = __wl_sym(lib, "wl_surface_interface", &ok);
        gWlLib.iRegion = __wl_sym(lib, "wl_region_interface", &ok);
        gWlLib.iShm = __wl_sym(lib, "wl_shm_interface", &ok);
        gWlLib.iShmPool = __wl_sym(lib, "wl_shm_pool_interface", &ok);
        gWlLib.iBuffer = __wl_sym(lib, "wl_buffer_interface", &ok);
        gWlLib.iSeat = __wl_sym(lib, "wl_seat_interface", &ok);
        gWlLib.iPointer = __wl_sym(lib, "wl_pointer_interface", &ok);
        gWlLib.iKeyboard = __wl_sym(lib, "wl_keyboard_interface", &ok);
        gWlLib.iOutput = __wl_sym(lib, "wl_output_interface", &ok);
        gWlLib.iDataDeviceManager = __wl_sym(lib, "wl_data_device_manager_interface", &ok);
        gWlLib.iDataDevice = __wl_sym(lib, "wl_data_device_interface", &ok);
        gWlLib.iDataSource = __wl_sym(lib, "wl_data_source_interface", &ok);
        gWlLib.iDataOffer = __wl_sym(lib, "wl_data_offer_interface", &ok);
        if (!ok) { return 0; }
        gWlLib.marshal_flags = (WlMarshalFlags_)dlsym(lib, "wl_proxy_marshal_array_flags");

        void* xkb = __wl_open2("libxkbcommon.so.0", "libxkbcommon.so");
        if (xkb != (void*)0) {
            int xo = 1;
            gWlLib.xkb_context_new = (XkbContextNew_)__wl_sym(xkb, "xkb_context_new", &xo);
            gWlLib.xkb_keymap_new_from_string = (XkbKeymapFromString_)__wl_sym(xkb, "xkb_keymap_new_from_string", &xo);
            gWlLib.xkb_keymap_unref = (XkbUnref_)__wl_sym(xkb, "xkb_keymap_unref", &xo);
            gWlLib.xkb_state_unref = (XkbUnref_)__wl_sym(xkb, "xkb_state_unref", &xo);
            gWlLib.xkb_state_new = (XkbStateNew_)__wl_sym(xkb, "xkb_state_new", &xo);
            gWlLib.xkb_state_update_mask = (XkbUpdateMask_)__wl_sym(xkb, "xkb_state_update_mask", &xo);
            gWlLib.xkb_state_key_get_one_sym = (XkbKeyU_)__wl_sym(xkb, "xkb_state_key_get_one_sym", &xo);
            gWlLib.xkb_state_key_get_utf32 = (XkbKeyU_)__wl_sym(xkb, "xkb_state_key_get_utf32", &xo);
            gWlLib.xkb_state_key_get_layout = (XkbKeyU_)__wl_sym(xkb, "xkb_state_key_get_layout", &xo);
            gWlLib.xkb_state_mod_name_is_active = (XkbModActive_)__wl_sym(xkb, "xkb_state_mod_name_is_active", &xo);
            gWlLib.xkb_keymap_key_get_syms_by_level = (XkbSymsByLevel_)__wl_sym(xkb, "xkb_keymap_key_get_syms_by_level", &xo);
            gWlLib.xkb_keymap_key_repeats = (XkbKeyRepeats_)__wl_sym(xkb, "xkb_keymap_key_repeats", &xo);
            gWlLib.xkb = xo;
            int co = 1;
            gWlLib.xkb_compose_table_new_from_locale = (XkbComposeTableNew_)__wl_sym(xkb, "xkb_compose_table_new_from_locale", &co);
            gWlLib.xkb_compose_state_new = (XkbComposeStateNew_)__wl_sym(xkb, "xkb_compose_state_new", &co);
            gWlLib.xkb_compose_state_feed = (XkbComposeFeed_)__wl_sym(xkb, "xkb_compose_state_feed", &co);
            gWlLib.xkb_compose_state_get_status = (XkbComposeStatus_)__wl_sym(xkb, "xkb_compose_state_get_status", &co);
            gWlLib.xkb_compose_state_get_one_sym = (XkbComposeKeysym_)__wl_sym(xkb, "xkb_compose_state_get_one_sym", &co);
            gWlLib.xkb_compose_state_reset = (XkbComposeReset_)__wl_sym(xkb, "xkb_compose_state_reset", &co);
            if (!co) { gWlLib.xkb_compose_table_new_from_locale = (XkbComposeTableNew_)0; }
        }
        void* cur = __wl_open2("libwayland-cursor.so.0", "libwayland-cursor.so");
        if (cur != (void*)0) {
            int co2 = 1;
            gWlLib.cursor_theme_load = (WlCursorThemeLoad_)__wl_sym(cur, "wl_cursor_theme_load", &co2);
            gWlLib.cursor_theme_destroy = (WlCursorThemeDestroy_)__wl_sym(cur, "wl_cursor_theme_destroy", &co2);
            gWlLib.cursor_theme_get_cursor = (WlCursorThemeGet_)__wl_sym(cur, "wl_cursor_theme_get_cursor", &co2);
            gWlLib.cursor_image_get_buffer = (WlCursorImageBuffer_)__wl_sym(cur, "wl_cursor_image_get_buffer", &co2);
            gWlLib.cursor = co2;
        }
    }
    gWlLib.loaded = 1;
    return 1;
}

// ── protocol tables built at run time ───────────────────────────────────

// struct wl_message / struct wl_interface (wayland-util.h).
struct GuiWlMessage {
    const char* name;
    const char* signature;
    const void** types;
};

struct GuiWlInterface {
    const char* name;
    int version;
    int methodCount;
    const GuiWlMessage* methods;
    int eventCount;
    const GuiWlMessage* events;
};

#define GUI_WL_MAX_MSG 16

struct GuiWlProto {
    struct GuiWlInterface iface;
    struct GuiWlMessage methods[GUI_WL_MAX_MSG];
    struct GuiWlMessage events[GUI_WL_MAX_MSG];
};

// No argument needs its interface listed: none of these messages carries
// a server-created object, so null types only skip libwayland's
// object-type check (and its debug output's interface names).
static const void* gWlNullTypes[8];

static struct GuiWlProto gWlXdgWmBase;
static struct GuiWlProto gWlXdgSurface;
static struct GuiWlProto gWlXdgToplevel;
static struct GuiWlProto gWlDecoManager;
static struct GuiWlProto gWlDeco;
static struct GuiWlProto gWlViewporter;
static struct GuiWlProto gWlViewport;
static struct GuiWlProto gWlFracManager;
static struct GuiWlProto gWlFrac;
static struct GuiWlProto gWlShapeManager;
static struct GuiWlProto gWlShapeDevice;

static void __wl_proto(&GuiWlProto p, const char* name, int version) {
    unsafe {
        memset((void*)p, 0, sizeof(struct GuiWlProto));
        p.iface.name = name;
        p.iface.version = version;
        p.iface.methods = &p.methods[0];
        p.iface.events = &p.events[0];
    }
}

static void __wl_req(&GuiWlProto p, const char* name, const char* sig) {
    unsafe {
        int i = p.iface.methodCount;
        p.methods[i].name = name;
        p.methods[i].signature = sig;
        p.methods[i].types = &gWlNullTypes[0];
        p.iface.methodCount = i + 1;
    }
}

static void __wl_evt(&GuiWlProto p, const char* name, const char* sig) {
    unsafe {
        int i = p.iface.eventCount;
        p.events[i].name = name;
        p.events[i].signature = sig;
        p.events[i].types = &gWlNullTypes[0];
        p.iface.eventCount = i + 1;
    }
}

static void __wl_build_protocols() {
    // xdg-shell (stable), the subset of version 6 used here
    __wl_proto(&gWlXdgWmBase, "xdg_wm_base", 6);
    __wl_req(&gWlXdgWmBase, "destroy", "");
    __wl_req(&gWlXdgWmBase, "create_positioner", "n");
    __wl_req(&gWlXdgWmBase, "get_xdg_surface", "no");
    __wl_req(&gWlXdgWmBase, "pong", "u");
    __wl_evt(&gWlXdgWmBase, "ping", "u");

    __wl_proto(&gWlXdgSurface, "xdg_surface", 6);
    __wl_req(&gWlXdgSurface, "destroy", "");
    __wl_req(&gWlXdgSurface, "get_toplevel", "n");
    __wl_req(&gWlXdgSurface, "get_popup", "n?oo");
    __wl_req(&gWlXdgSurface, "set_window_geometry", "iiii");
    __wl_req(&gWlXdgSurface, "ack_configure", "u");
    __wl_evt(&gWlXdgSurface, "configure", "u");

    __wl_proto(&gWlXdgToplevel, "xdg_toplevel", 6);
    __wl_req(&gWlXdgToplevel, "destroy", "");
    __wl_req(&gWlXdgToplevel, "set_parent", "?o");
    __wl_req(&gWlXdgToplevel, "set_title", "s");
    __wl_req(&gWlXdgToplevel, "set_app_id", "s");
    __wl_req(&gWlXdgToplevel, "show_window_menu", "ouii");
    __wl_req(&gWlXdgToplevel, "move", "ou");
    __wl_req(&gWlXdgToplevel, "resize", "ouu");
    __wl_req(&gWlXdgToplevel, "set_max_size", "ii");
    __wl_req(&gWlXdgToplevel, "set_min_size", "ii");
    __wl_req(&gWlXdgToplevel, "set_maximized", "");
    __wl_req(&gWlXdgToplevel, "unset_maximized", "");
    __wl_req(&gWlXdgToplevel, "set_fullscreen", "?o");
    __wl_req(&gWlXdgToplevel, "unset_fullscreen", "");
    __wl_req(&gWlXdgToplevel, "set_minimized", "");
    __wl_evt(&gWlXdgToplevel, "configure", "iia");
    __wl_evt(&gWlXdgToplevel, "close", "");
    __wl_evt(&gWlXdgToplevel, "configure_bounds", "4ii");
    __wl_evt(&gWlXdgToplevel, "wm_capabilities", "5a");

    // xdg-decoration-unstable-v1
    __wl_proto(&gWlDecoManager, "zxdg_decoration_manager_v1", 1);
    __wl_req(&gWlDecoManager, "destroy", "");
    __wl_req(&gWlDecoManager, "get_toplevel_decoration", "no");
    __wl_proto(&gWlDeco, "zxdg_toplevel_decoration_v1", 1);
    __wl_req(&gWlDeco, "destroy", "");
    __wl_req(&gWlDeco, "set_mode", "u");
    __wl_req(&gWlDeco, "unset_mode", "");
    __wl_evt(&gWlDeco, "configure", "u");

    // viewporter (stable)
    __wl_proto(&gWlViewporter, "wp_viewporter", 1);
    __wl_req(&gWlViewporter, "destroy", "");
    __wl_req(&gWlViewporter, "get_viewport", "no");
    __wl_proto(&gWlViewport, "wp_viewport", 1);
    __wl_req(&gWlViewport, "destroy", "");
    __wl_req(&gWlViewport, "set_source", "ffff");
    __wl_req(&gWlViewport, "set_destination", "ii");

    // fractional-scale-v1 (staging)
    __wl_proto(&gWlFracManager, "wp_fractional_scale_manager_v1", 1);
    __wl_req(&gWlFracManager, "destroy", "");
    __wl_req(&gWlFracManager, "get_fractional_scale", "no");
    __wl_proto(&gWlFrac, "wp_fractional_scale_v1", 1);
    __wl_req(&gWlFrac, "destroy", "");
    __wl_evt(&gWlFrac, "preferred_scale", "u");

    // cursor-shape-v1 (staging)
    __wl_proto(&gWlShapeManager, "wp_cursor_shape_manager_v1", 1);
    __wl_req(&gWlShapeManager, "destroy", "");
    __wl_req(&gWlShapeManager, "get_pointer", "no");
    __wl_req(&gWlShapeManager, "get_tablet_tool_v2", "no");
    __wl_proto(&gWlShapeDevice, "wp_cursor_shape_device_v1", 1);
    __wl_req(&gWlShapeDevice, "destroy", "");
    __wl_req(&gWlShapeDevice, "set_shape", "uu");
}

// ── marshalling ─────────────────────────────────────────────────────────

// union wl_argument, 8 bytes: an argument list is an array of these.
struct GuiWlArgs {
    unsigned long long a[8];
};

static struct GuiWlArgs __wl_args() {
    struct GuiWlArgs r;
    unsafe { memset((void*)&r, 0, sizeof(struct GuiWlArgs)); }
    return r;
}
static void __wl_set_i(&GuiWlArgs r, int i, int v) { unsafe { *(int*)&r.a[i] = v; } }
static void __wl_set_u(&GuiWlArgs r, int i, unsigned int v) { unsafe { *(unsigned int*)&r.a[i] = v; } }
static void __wl_set_p(&GuiWlArgs r, int i, const void* p) { unsafe { *(const void**)&r.a[i] = p; } }

static int __wl_arg_i(void* args, int i) { unsafe { return *(int*)((unsigned long)args + (unsigned long)i * 8UL); } }
static unsigned int __wl_arg_u(void* args, int i) { unsafe { return *(unsigned int*)((unsigned long)args + (unsigned long)i * 8UL); } }
static void* __wl_arg_p(void* args, int i) { unsafe { return *(void**)((unsigned long)args + (unsigned long)i * 8UL); } }
static double __wl_arg_fixed(void* args, int i) { return (double)__wl_arg_i(args, i) / 256.0; }

// Sends request 'opcode' on 'proxy'. 'iface' (non-null) makes the request
// create a new object of that interface — its 'n' argument — and returns
// it. 'destroy' destroys the proxy afterwards (destructor requests).
static void* __wl_send(void* proxy, unsigned int opcode, const void* iface, unsigned int version, &GuiWlArgs args, int destroy) {
    unsafe {
        if (proxy == (void*)0) { return (void*)0; }
        unsigned int ver = version;
        if (ver == 0U) { ver = gWlLib.get_version(proxy); }
        if (gWlLib.marshal_flags != (WlMarshalFlags_)0) {
            return gWlLib.marshal_flags(proxy, opcode, iface, ver, destroy ? 1U : 0U, (void*)&args.a[0]);
        }
        void* r = gWlLib.marshal_ctor(proxy, opcode, (void*)&args.a[0], iface, ver);
        if (destroy) { gWlLib.proxy_destroy(proxy); }
        return r;
    }
}

static void __wl_req0(void* proxy, unsigned int opcode) {
    struct GuiWlArgs a = __wl_args();
    __wl_send(proxy, opcode, (const void*)0, 0U, &a, 0);
}

static void __wl_destroy_req(void* proxy, unsigned int opcode) {
    unsafe {
        if (proxy == (void*)0) { return; }
        struct GuiWlArgs a = __wl_args();
        __wl_send(proxy, opcode, (const void*)0, 0U, &a, 1);
    }
}

// A request whose first argument is the new object ('n...'), the rest
// filled in by the caller from index 1.
static void* __wl_new(void* proxy, unsigned int opcode, const void* iface, &GuiWlArgs a) {
    return __wl_send(proxy, opcode, iface, 0U, a, 0);
}

static int __wl_streq(const char* a, const char* b) {
    unsafe {
        if (a == (const char*)0 || b == (const char*)0) { return 0; }
        unsigned long i = 0UL;
        while (a[i] != '\0' && a[i] == b[i]) { i = i + 1UL; }
        return a[i] == b[i] ? 1 : 0;
    }
}

static int __wl_iface_version(const void* iface) {
    unsafe { return ((const struct GuiWlInterface*)iface)->version; }
}

// ── state ───────────────────────────────────────────────────────────────

#define GUI_WL_MAX_BUFFERS 4
#define GUI_WL_MAX_OUTPUTS 16

struct GuiWlBuffer {
    void* buffer;            // wl_buffer
    unsigned char* data;     // mmap()ed pixels, BGRA (WL_SHM_FORMAT_ARGB8888)
    unsigned long size;
    int w; int h;
    int busy;                // attached and not yet released by the compositor
    int orphan;              // replaced (size changed): destroy on release
    int dx0; int dy0; int dx1; int dy1;   // stale region (empty when dx1 <= dx0)
};

struct GuiWlWindow {
    struct GuiWindowState* st;
    void* surface;
    void* xdgSurface;
    void* toplevel;
    void* decoration;
    void* viewport;
    void* fracScale;
    void* frameCb;
    double frameAt;          // ms when frameCb was requested
    int configured;          // initial configure seen (the surface may show buffers)
    int created;             // window_create returned
    int cfgW; int cfgH;      // the last xdg_toplevel.configure size (0 = ours to pick)
    int intScale;            // wl_surface buffer scale, when there is no viewport
    int fracScale120;        // preferred fractional scale * 120 (0 = none yet)
    int prefBufScale;        // wl_surface.preferred_buffer_scale (0 = none yet)
    void* outputs[GUI_WL_MAX_OUTPUTS];   // outputs the surface is on
    struct GuiWlBuffer* bufs[GUI_WL_MAX_BUFFERS];
    struct GuiWlBuffer* pending;  // filled, waiting for the frame callback to be committed
    int pdx0; int pdy0; int pdx1; int pdy1;  // damage of 'pending'
    int hasContent;          // a buffer is attached (the surface is mapped)
    int decorated;
};

struct GuiWlOutput {
    void* proxy;
    unsigned int name;
    int scale;
};

struct GuiWlOffer {
    void* proxy;
    int utf8;      // text/plain;charset=utf-8
    int plain;     // text/plain
    int x11utf8;   // UTF8_STRING
    int string;    // STRING / TEXT
};

#define GUI_WL_MAX_OFFERS 8

struct GuiWl {
    void* display;
    void* registry;
    void* compositor; int compositorVer;
    void* shm;
    void* wmBase;
    void* seat; unsigned int seatName; int seatVer;
    void* pointer; int pointerVer;
    void* keyboard;
    void* ddm;          // wl_data_device_manager
    void* dataDevice;
    void* decoManager;
    void* viewporter;
    void* fracManager;
    void* shapeManager;
    void* shapeDevice;
    struct GuiWlOutput outputs[GUI_WL_MAX_OUTPUTS];
    int outputCount;
    int wakeR; int wakeW;
    int dead;
    double envScale;    // $SAFEC_GUI_SCALE, 0 = none

    // input
    unsigned long long ptrFocus;   // surface key the pointer is in
    unsigned long long kbFocus;
    unsigned int ptrSerial;        // the pointer enter serial (for set_cursor)
    unsigned int lastSerial;       // latest input serial (for the clipboard)
    double ptrX; double ptrY;      // surface coordinates
    int buttons;
    int mods;
    double axisDx; double axisDy;  // accumulated within one wl_pointer.frame
    int axisDiscX; int axisDiscY;  // value120 units (discrete steps * 120)
    int axisHave; int axisHaveDisc;
    // keyboard
    void* xkbCtx; void* keymap; void* xkbState;
    void* compose;                 // xkb_compose_state, or null
    int repeatRate; int repeatDelay;
    unsigned int repeatKey;        // evdev code + 8, 0 = none
    double repeatNext;
    struct GuiEvent repeatEv;
    // cursors
    void* cursorTheme; int cursorThemeSize;
    void* cursorSurface;
    int cursorShown;               // GUI_CURSOR_* currently on the pointer
    // clipboard
    struct GuiWlOffer offers[GUI_WL_MAX_OFFERS];
    int selection;                 // index into offers, -1 = none
    void* source;                  // our wl_data_source while we own the selection
    struct String clip;
    char appId[64];
};

static struct GuiWl gWl;
static int gWlState = 0;     // 0 not tried, 1 connected, -1 unavailable
static struct GuiBackend gWlBackend;
static int gWlBackendInit = 0;

static double __wl_now_ms() { return (double)time_mono_ns() / 1000000.0; }

static struct GuiWlWindow* __wl_win(&GuiWindowState st) {
    unsafe { return (struct GuiWlWindow*)st.native; }
}

static struct GuiWindowState* __wl_find(void* surface) {
    unsafe {
        if (surface == (void*)0) { return (struct GuiWindowState*)0; }
        ?&GuiWindowState o = gui_backend_find((unsigned long long)(unsigned long)surface);
        if (o.is_null()) { return (struct GuiWindowState*)0; }
        return (struct GuiWindowState*)o;
    }
}

// ── scale ───────────────────────────────────────────────────────────────

static int __wl_output_scale(void* proxy) {
    unsafe {
        int i = 0;
        while (i < gWl.outputCount) {
            if (gWl.outputs[i].proxy == proxy) { return gWl.outputs[i].scale; }
            i = i + 1;
        }
    }
    return 1;
}

static double __wl_display_scale() {
    if (gWl.envScale > 0.0) { return gWl.envScale; }
    int s = 1;
    int i = 0;
    while (i < gWl.outputCount) {
        if (gWl.outputs[i].scale > s) { s = gWl.outputs[i].scale; }
        i = i + 1;
    }
    return (double)s;
}

// The scale a window's buffer should have now.
static double __wl_window_scale(struct GuiWlWindow* w) {
    unsafe {
        if (!w->st->hiDpi) { return 1.0; }
        if (gWl.envScale > 0.0) { return gWl.envScale; }
        if (w->fracScale120 > 0 && gWl.viewporter != (void*)0) { return (double)w->fracScale120 / 120.0; }
        if (w->prefBufScale > 0) { return (double)w->prefBufScale; }
        int s = 0;
        int i = 0;
        while (i < GUI_WL_MAX_OUTPUTS) {
            if (w->outputs[i] != (void*)0) {
                int os = __wl_output_scale(w->outputs[i]);
                if (os > s) { s = os; }
            }
            i = i + 1;
        }
        if (s == 0) { return __wl_display_scale(); }
        return (double)s;
    }
}

static int __wl_round(double v) { return (int)(v + 0.5); }

// Applies the surface-side scaling state for the window's current scale:
// with a viewport the buffer is drawn at the logical size (any scale);
// without one, an integer buffer scale.
static void __wl_apply_scaling(struct GuiWlWindow* w) {
    unsafe {
        struct GuiWindowState* st = w->st;
        if (w->viewport != (void*)0) {
            struct GuiWlArgs a = __wl_args();
            __wl_set_i(&a, 0, st->logicalW);
            __wl_set_i(&a, 1, st->logicalH);
            __wl_send(w->viewport, 2U /* set_destination */, (const void*)0, 0U, &a, 0);
        } else if (gWl.compositorVer >= 3) {
            int s = (int)(st->scale + 0.5);
            if (s < 1) { s = 1; }
            w->intScale = s;
            // double-buffered: applies with the next commit, whoever makes
            // it (the shm path, or a GPU presenter's eglSwapBuffers)
            struct GuiWlArgs b = __wl_args();
            __wl_set_i(&b, 0, s);
            __wl_send(w->surface, 8U /* set_buffer_scale */, (const void*)0, 0U, &b, 0);
        }
    }
}

// Re-evaluates the window's scale; on a change updates the buffer size and
// posts GUI_EVENT_SCALE.
static void __wl_rescale(struct GuiWlWindow* w) {
    unsafe {
        struct GuiWindowState* st = w->st;
        double s = __wl_window_scale(w);
        if (w->viewport == (void*)0) {
            // integer buffer scales only
            s = (double)(int)(s + 0.999);
            if (s < 1.0) { s = 1.0; }
        }
        if (s == st->scale) { return; }
        st->scale = s;
        st->pixelW = __wl_round((double)st->logicalW * s);
        st->pixelH = __wl_round((double)st->logicalH * s);
        __wl_apply_scaling(w);
        if (w->created) {
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCALE);
            ev.width = st->pixelW;
            ev.height = st->pixelH;
            gui_backend_post(st, ev);
        }
    }
}

static void __wl_set_logical(struct GuiWlWindow* w, int lw, int lh) {
    unsafe {
        struct GuiWindowState* st = w->st;
        if (st->minW > 0 && lw < st->minW) { lw = st->minW; }
        if (st->minH > 0 && lh < st->minH) { lh = st->minH; }
        if (st->maxW > 0 && lw > st->maxW) { lw = st->maxW; }
        if (st->maxH > 0 && lh > st->maxH) { lh = st->maxH; }
        if (lw < 1) { lw = 1; }
        if (lh < 1) { lh = 1; }
        if (lw == st->logicalW && lh == st->logicalH) { return; }
        st->logicalW = lw;
        st->logicalH = lh;
        st->pixelW = __wl_round((double)lw * st->scale);
        st->pixelH = __wl_round((double)lh * st->scale);
        __wl_apply_scaling(w);
        if (w->created) {
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_RESIZE);
            ev.width = st->pixelW;
            ev.height = st->pixelH;
            gui_backend_post(st, ev);
        }
    }
}

// ── shm buffers ─────────────────────────────────────────────────────────

typedef fn int(const char*, unsigned int) MemfdCreate_;

static int __wl_shm_fd(unsigned long size) {
    unsafe {
        int fd = -1;
        MemfdCreate_ mfd = (MemfdCreate_)dlsym((void*)0 /* RTLD_DEFAULT */, "memfd_create");
        if (mfd != (MemfdCreate_)0) { fd = mfd("safec-gui", 1U /* MFD_CLOEXEC */); }
        if (fd < 0) {
            char path[256];
            const char* dir = getenv("XDG_RUNTIME_DIR");
            if (dir == (const char*)0) { dir = "/tmp"; }
            unsigned long n = 0UL;
            while (dir[n] != '\0' && n < 200UL) { path[n] = dir[n]; n = n + 1UL; }
            const char* tail = "/safec-gui-XXXXXX";
            unsigned long k = 0UL;
            while (tail[k] != '\0') { path[n + k] = tail[k]; k = k + 1UL; }
            path[n + k] = '\0';
            fd = mkstemp(&path[0]);
            if (fd >= 0) { unlink(&path[0]); }
        }
        if (fd < 0) { return -1; }
        if (ftruncate(fd, (long)size) != 0) { close(fd); return -1; }
        return fd;
    }
}

static void __wl_buffer_free(struct GuiWlBuffer* b) {
    unsafe {
        if (b == (struct GuiWlBuffer*)0) { return; }
        if (b->buffer != (void*)0) { __wl_destroy_req(b->buffer, 0U /* destroy */); }
        if (b->data != (unsigned char*)0) { munmap((void*)b->data, b->size); }
        dealloc_raw_((void*)b);
    }
}

static int __wl_buffer_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlBuffer* b = (struct GuiWlBuffer*)data;
        if (opcode == 0U) {   // release
            b->busy = 0;
            if (b->orphan) { __wl_buffer_free(b); }
        }
    }
    return 0;
}

static struct GuiWlBuffer* __wl_buffer_new(int w, int h) {
    unsafe {
        unsigned long stride = (unsigned long)w * 4UL;
        unsigned long size = stride * (unsigned long)h;
        int fd = __wl_shm_fd(size);
        if (fd < 0) { return (struct GuiWlBuffer*)0; }
        void* data = mmap((void*)0, size, 3 /* PROT_READ|PROT_WRITE */, 1 /* MAP_SHARED */, fd, 0L);
        if (data == (void*)0 || (long)data == -1L) { close(fd); return (struct GuiWlBuffer*)0; }
        struct GuiWlArgs a = __wl_args();
        __wl_set_i(&a, 1, fd);
        __wl_set_i(&a, 2, (int)size);
        void* pool = __wl_new(gWl.shm, 0U /* create_pool */, gWlLib.iShmPool, &a);
        close(fd);
        if (pool == (void*)0) { munmap(data, size); return (struct GuiWlBuffer*)0; }
        struct GuiWlArgs b = __wl_args();
        __wl_set_i(&b, 1, 0);
        __wl_set_i(&b, 2, w);
        __wl_set_i(&b, 3, h);
        __wl_set_i(&b, 4, (int)stride);
        __wl_set_u(&b, 5, 0U /* WL_SHM_FORMAT_ARGB8888 */);
        void* buffer = __wl_new(pool, 0U /* create_buffer */, gWlLib.iBuffer, &b);
        __wl_destroy_req(pool, 1U /* destroy */);
        if (buffer == (void*)0) { munmap(data, size); return (struct GuiWlBuffer*)0; }
        struct GuiWlBuffer* r = (struct GuiWlBuffer*)alloc_zeroed_raw_(sizeof(struct GuiWlBuffer));
        r->buffer = buffer;
        r->data = (unsigned char*)data;
        r->size = size;
        r->w = w;
        r->h = h;
        r->dx0 = 0; r->dy0 = 0; r->dx1 = w; r->dy1 = h;   // all stale
        gWlLib.add_dispatcher(buffer, (void*)__wl_buffer_dispatch, (const void*)r, (void*)0);
        return r;
    }
}

// ── windows ─────────────────────────────────────────────────────────────

static void __wl_toplevel_limits(struct GuiWlWindow* w) {
    unsafe {
        struct GuiWindowState* st = w->st;
        if (w->toplevel == (void*)0) { return; }
        int minW = st->minW; int minH = st->minH; int maxW = st->maxW; int maxH = st->maxH;
        if (!st->resizable) {
            minW = st->logicalW; maxW = st->logicalW;
            minH = st->logicalH; maxH = st->logicalH;
        }
        struct GuiWlArgs a = __wl_args();
        __wl_set_i(&a, 0, maxW > 0 ? maxW : 0);
        __wl_set_i(&a, 1, maxH > 0 ? maxH : 0);
        __wl_send(w->toplevel, 7U /* set_max_size */, (const void*)0, 0U, &a, 0);
        struct GuiWlArgs b = __wl_args();
        __wl_set_i(&b, 0, minW > 0 ? minW : 0);
        __wl_set_i(&b, 1, minH > 0 ? minH : 0);
        __wl_send(w->toplevel, 8U /* set_min_size */, (const void*)0, 0U, &b, 0);
    }
}

static void __wl_commit(struct GuiWlWindow* w) {
    unsafe { __wl_req0(w->surface, 6U /* commit */); }
}

static void __wl_frame_request(struct GuiWlWindow* w);
static void __wl_commit_pending(struct GuiWlWindow* w);

static int __wl_callback_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlWindow* w = (struct GuiWlWindow*)data;
        if (opcode == 0U && w->frameCb == target) {   // done
            gWlLib.proxy_destroy(target);
            w->frameCb = (void*)0;
            if (w->pending != (struct GuiWlBuffer*)0) { __wl_commit_pending(w); }
        }
    }
    return 0;
}

static void __wl_frame_request(struct GuiWlWindow* w) {
    unsafe {
        struct GuiWlArgs a = __wl_args();
        void* cb = __wl_new(w->surface, 3U /* frame */, gWlLib.iCallback, &a);
        if (cb != (void*)0) {
            gWlLib.add_dispatcher(cb, (void*)__wl_callback_dispatch, (const void*)w, (void*)0);
            w->frameCb = cb;
            w->frameAt = __wl_now_ms();
        }
    }
}

// Attaches and commits the filled buffer 'pending' with its damage.
static void __wl_commit_pending(struct GuiWlWindow* w) {
    unsafe {
        struct GuiWlBuffer* b = w->pending;
        if (b == (struct GuiWlBuffer*)0 || !w->configured) { return; }
        w->pending = (struct GuiWlBuffer*)0;
        struct GuiWlArgs a = __wl_args();
        __wl_set_p(&a, 0, b->buffer);
        __wl_send(w->surface, 1U /* attach */, (const void*)0, 0U, &a, 0);
        if (w->viewport == (void*)0 && gWl.compositorVer >= 3) {
            struct GuiWlArgs s = __wl_args();
            __wl_set_i(&s, 0, w->intScale > 0 ? w->intScale : 1);
            __wl_send(w->surface, 8U /* set_buffer_scale */, (const void*)0, 0U, &s, 0);
        }
        struct GuiWlArgs d = __wl_args();
        if (gWl.compositorVer >= 4) {
            __wl_set_i(&d, 0, w->pdx0);
            __wl_set_i(&d, 1, w->pdy0);
            __wl_set_i(&d, 2, w->pdx1 - w->pdx0);
            __wl_set_i(&d, 3, w->pdy1 - w->pdy0);
            __wl_send(w->surface, 9U /* damage_buffer */, (const void*)0, 0U, &d, 0);
        } else {
            __wl_set_i(&d, 0, 0);
            __wl_set_i(&d, 1, 0);
            __wl_set_i(&d, 2, 0x7FFFFFFF);
            __wl_set_i(&d, 3, 0x7FFFFFFF);
            __wl_send(w->surface, 2U /* damage */, (const void*)0, 0U, &d, 0);
        }
        b->busy = 1;
        w->hasContent = 1;
        __wl_frame_request(w);
        __wl_commit(w);
        gWlLib.display_flush(gWl.display);
    }
}

static void __wl_xdg_surface_configure(struct GuiWlWindow* w, unsigned int serial) {
    unsafe {
        struct GuiWlArgs a = __wl_args();
        __wl_set_u(&a, 0, serial);
        __wl_send(w->xdgSurface, 4U /* ack_configure */, (const void*)0, 0U, &a, 0);
        int first = w->configured ? 0 : 1;
        w->configured = 1;
        int lw = w->st->logicalW;
        int lh = w->st->logicalH;
        if (w->cfgW > 0 && w->cfgH > 0) { __wl_set_logical(w, w->cfgW, w->cfgH); }
        // Same size (a state change: focus, maximize at the same size...):
        // commit at once so a compositor that waits for the acked state
        // (sway's transactions) isn't kept waiting for a frame that the
        // application has no reason to draw.
        if (!first && lw == w->st->logicalW && lh == w->st->logicalH) {
            __wl_commit(w);
            gWlLib.display_flush(gWl.display);
        }
        if (first && w->created) {
            // shown again (after hide) or first map of a window created hidden
            gui_backend_post(w->st, GuiEvent::new(GUI_EVENT_EXPOSE));
        }
    }
}

static int __wl_xdg_surface_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlWindow* w = (struct GuiWlWindow*)data;
        if (opcode == 0U) { __wl_xdg_surface_configure(w, __wl_arg_u(args, 0)); }
    }
    return 0;
}

static int __wl_toplevel_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlWindow* w = (struct GuiWlWindow*)data;
        if (opcode == 0U) {          // configure(width, height, states)
            w->cfgW = __wl_arg_i(args, 0);
            w->cfgH = __wl_arg_i(args, 1);
        } else if (opcode == 1U) {   // close
            if (w->created) { gui_backend_post(w->st, GuiEvent::new(GUI_EVENT_CLOSE)); }
        }
    }
    return 0;
}

static int __wl_frac_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlWindow* w = (struct GuiWlWindow*)data;
        if (opcode == 0U) {          // preferred_scale (120ths)
            w->fracScale120 = (int)__wl_arg_u(args, 0);
            __wl_rescale(w);
        }
    }
    return 0;
}

static int __wl_surface_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        struct GuiWlWindow* w = (struct GuiWlWindow*)data;
        if (opcode == 0U || opcode == 1U) {   // enter / leave (output)
            void* out = __wl_arg_p(args, 0);
            int i = 0;
            int slot = -1;
            while (i < GUI_WL_MAX_OUTPUTS) {
                if (w->outputs[i] == out) { slot = i; }
                i = i + 1;
            }
            if (opcode == 0U && slot < 0) {
                i = 0;
                while (i < GUI_WL_MAX_OUTPUTS) {
                    if (w->outputs[i] == (void*)0) { w->outputs[i] = out; break; }
                    i = i + 1;
                }
            }
            if (opcode == 1U && slot >= 0) { w->outputs[slot] = (void*)0; }
            __wl_rescale(w);
        } else if (opcode == 2U) {   // preferred_buffer_scale
            w->prefBufScale = __wl_arg_i(args, 0);
            __wl_rescale(w);
        }
    }
    return 0;
}

static int __wl_deco_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    return 0;
}

// The program's name for xdg_toplevel.set_app_id: $SAFEC_GUI_APP_ID, else
// /proc/self/comm.
static void __wl_init_app_id() {
    unsafe {
        gWl.appId[0] = '\0';
        const char* env = getenv("SAFEC_GUI_APP_ID");
        unsigned long n = 0UL;
        if (env != (const char*)0) {
            while (env[n] != '\0' && n < 63UL) { gWl.appId[n] = env[n]; n = n + 1UL; }
            gWl.appId[n] = '\0';
            return;
        }
        int fd = open("/proc/self/comm", 0 /* O_RDONLY */);
        if (fd >= 0) {
            long r = read(fd, (void*)&gWl.appId[0], 63UL);
            close(fd);
            if (r > 0L) {
                n = (unsigned long)r;
                while (n > 0UL && (gWl.appId[n - 1UL] == '\n' || gWl.appId[n - 1UL] == ' ')) { n = n - 1UL; }
                gWl.appId[n] = '\0';
            }
        }
        if (gWl.appId[0] == '\0') {
            const char* d = "safec";
            n = 0UL;
            while (d[n] != '\0') { gWl.appId[n] = d[n]; n = n + 1UL; }
            gWl.appId[n] = '\0';
        }
    }
}

static void __wl_set_title(&GuiWindowState st, const &char title) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0 || w->toplevel == (void*)0) { return; }
        struct GuiWlArgs a = __wl_args();
        __wl_set_p(&a, 0, (const void*)title);
        __wl_send(w->toplevel, 2U /* set_title */, (const void*)0, 0U, &a, 0);
        gWlLib.display_flush(gWl.display);
    }
}

// Waits (dispatching) until the window's initial configure arrives.
static void __wl_wait_configure(struct GuiWlWindow* w) {
    unsafe {
        int tries = 0;
        while (!w->configured && tries < 20 && !gWl.dead) {
            if (gWlLib.display_roundtrip(gWl.display) < 0) { gWl.dead = 1; break; }
            tries = tries + 1;
        }
    }
}

static void __wl_map(struct GuiWlWindow* w) {
    unsafe {
        // The toplevel's state (an unmapped surface starts over without it),
        // then the initial commit — no buffer; the compositor answers with
        // the first configure.
        struct GuiWindowState* st = w->st;
        struct GuiWlArgs t = __wl_args();
        __wl_set_p(&t, 0, (const void*)st->title.data);
        __wl_send(w->toplevel, 2U /* set_title */, (const void*)0, 0U, &t, 0);
        struct GuiWlArgs id = __wl_args();
        __wl_set_p(&id, 0, (const void*)&gWl.appId[0]);
        __wl_send(w->toplevel, 3U /* set_app_id */, (const void*)0, 0U, &id, 0);
        __wl_toplevel_limits(w);
        if (w->decoration != (void*)0) {
            struct GuiWlArgs m = __wl_args();
            __wl_set_u(&m, 0, w->decorated ? 2U /* server_side */ : 1U /* client_side */);
            __wl_send(w->decoration, 1U /* set_mode */, (const void*)0, 0U, &m, 0);
        }
        w->configured = 0;
        __wl_commit(w);
        gWlLib.display_flush(gWl.display);
        __wl_wait_configure(w);
    }
}

static int __wl_create(&GuiWindowState st, const &GuiWindowOptions opts) {
    unsafe {
        if (gWl.compositor == (void*)0 || gWl.wmBase == (void*)0 || gWl.shm == (void*)0 || gWl.dead) { return 0; }
        struct GuiWlWindow* w = (struct GuiWlWindow*)alloc_zeroed_raw_(sizeof(struct GuiWlWindow));
        w->st = (struct GuiWindowState*)st;
        struct GuiWlArgs a = __wl_args();
        w->surface = __wl_new(gWl.compositor, 0U /* create_surface */, gWlLib.iSurface, &a);
        if (w->surface == (void*)0) { dealloc_raw_((void*)w); return 0; }
        gWlLib.add_dispatcher(w->surface, (void*)__wl_surface_dispatch, (const void*)w, (void*)w);

        struct GuiWlArgs b = __wl_args();
        __wl_set_p(&b, 1, w->surface);
        w->xdgSurface = __wl_new(gWl.wmBase, 2U /* get_xdg_surface */, (const void*)&gWlXdgSurface.iface, &b);
        gWlLib.add_dispatcher(w->xdgSurface, (void*)__wl_xdg_surface_dispatch, (const void*)w, (void*)0);
        struct GuiWlArgs c = __wl_args();
        w->toplevel = __wl_new(w->xdgSurface, 1U /* get_toplevel */, (const void*)&gWlXdgToplevel.iface, &c);
        gWlLib.add_dispatcher(w->toplevel, (void*)__wl_toplevel_dispatch, (const void*)w, (void*)0);

        if (gWl.viewporter != (void*)0) {
            struct GuiWlArgs v = __wl_args();
            __wl_set_p(&v, 1, w->surface);
            w->viewport = __wl_new(gWl.viewporter, 1U /* get_viewport */, (const void*)&gWlViewport.iface, &v);
            if (gWl.fracManager != (void*)0 && opts.hiDpi) {
                struct GuiWlArgs f = __wl_args();
                __wl_set_p(&f, 1, w->surface);
                w->fracScale = __wl_new(gWl.fracManager, 1U /* get_fractional_scale */, (const void*)&gWlFrac.iface, &f);
                gWlLib.add_dispatcher(w->fracScale, (void*)__wl_frac_dispatch, (const void*)w, (void*)0);
            }
        }
        if (gWl.decoManager != (void*)0) {
            struct GuiWlArgs d = __wl_args();
            __wl_set_p(&d, 1, w->toplevel);
            w->decoration = __wl_new(gWl.decoManager, 1U /* get_toplevel_decoration */, (const void*)&gWlDeco.iface, &d);
            gWlLib.add_dispatcher(w->decoration, (void*)__wl_deco_dispatch, (const void*)w, (void*)0);
        }

        st.native = (void*)w;
        st.key = (unsigned long long)(unsigned long)w->surface;
        st.scale = 1.0;
        st.pixelW = st.logicalW;
        st.pixelH = st.logicalH;
        w->intScale = 1;
        __wl_rescale(w);

        w->decorated = opts.decorated;
        __wl_apply_scaling(w);
        if (opts.visible) { __wl_map(w); }
        if (gWl.dead) {
            st.native = null;
            dealloc_raw_((void*)w);
            return 0;
        }
        // st.x/st.y: Wayland never tells a client where its window is.
        w->created = 1;
        return 1;
    }
}

static void __wl_window_free_buffers(struct GuiWlWindow* w) {
    unsafe {
        int i = 0;
        while (i < GUI_WL_MAX_BUFFERS) {
            if (w->bufs[i] != (struct GuiWlBuffer*)0) {
                __wl_buffer_free(w->bufs[i]);
                w->bufs[i] = (struct GuiWlBuffer*)0;
            }
            i = i + 1;
        }
        w->pending = (struct GuiWlBuffer*)0;
    }
}

static void __wl_destroy(&GuiWindowState st) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0) { return; }
        if (gWl.ptrFocus == st.key) { gWl.ptrFocus = 0ULL; }
        if (gWl.kbFocus == st.key) { gWl.kbFocus = 0ULL; gWl.repeatKey = 0U; }
        if (w->frameCb != (void*)0) { gWlLib.proxy_destroy(w->frameCb); }
        if (w->fracScale != (void*)0) { __wl_destroy_req(w->fracScale, 0U); }
        if (w->viewport != (void*)0) { __wl_destroy_req(w->viewport, 0U); }
        if (w->decoration != (void*)0) { __wl_destroy_req(w->decoration, 0U); }
        if (w->toplevel != (void*)0) { __wl_destroy_req(w->toplevel, 0U); }
        if (w->xdgSurface != (void*)0) { __wl_destroy_req(w->xdgSurface, 0U); }
        __wl_destroy_req(w->surface, 0U);
        __wl_window_free_buffers(w);
        gWlLib.display_flush(gWl.display);
        dealloc_raw_((void*)w);
        st.native = null;
    }
}

static void __wl_add_damage(struct GuiWlBuffer* b, int x0, int y0, int x1, int y1) {
    unsafe {
        if (b->dx1 <= b->dx0 || b->dy1 <= b->dy0) {
            b->dx0 = x0; b->dy0 = y0; b->dx1 = x1; b->dy1 = y1;
            return;
        }
        if (x0 < b->dx0) { b->dx0 = x0; }
        if (y0 < b->dy0) { b->dy0 = y0; }
        if (x1 > b->dx1) { b->dx1 = x1; }
        if (y1 > b->dy1) { b->dy1 = y1; }
    }
}

// A buffer of bw x bh to draw the next frame into: the pending one if a
// frame is already waiting, else a free one (made if needed).
static struct GuiWlBuffer* __wl_next_buffer(struct GuiWlWindow* w, int bw, int bh) {
    unsafe {
        if (w->pending != (struct GuiWlBuffer*)0) {
            if (w->pending->w == bw && w->pending->h == bh) { return w->pending; }
            w->pending = (struct GuiWlBuffer*)0;
        }
        int i = 0;
        // drop free buffers of another size; orphan busy ones
        while (i < GUI_WL_MAX_BUFFERS) {
            struct GuiWlBuffer* b = w->bufs[i];
            if (b != (struct GuiWlBuffer*)0 && (b->w != bw || b->h != bh)) {
                if (b->busy) { b->orphan = 1; } else { __wl_buffer_free(b); }
                w->bufs[i] = (struct GuiWlBuffer*)0;
            }
            i = i + 1;
        }
        int attempt = 0;
        while (attempt < 50) {
            i = 0;
            while (i < GUI_WL_MAX_BUFFERS) {
                struct GuiWlBuffer* b = w->bufs[i];
                if (b != (struct GuiWlBuffer*)0 && !b->busy) { return b; }
                i = i + 1;
            }
            i = 0;
            while (i < GUI_WL_MAX_BUFFERS) {
                if (w->bufs[i] == (struct GuiWlBuffer*)0) {
                    w->bufs[i] = __wl_buffer_new(bw, bh);
                    return w->bufs[i];
                }
                i = i + 1;
            }
            // All held by the compositor: wait for a release.
            if (gWlLib.display_roundtrip(gWl.display) < 0) { gWl.dead = 1; break; }
            attempt = attempt + 1;
        }
        return (struct GuiWlBuffer*)0;
    }
}

static void __wl_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                         int x, int y, int rw, int rh) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0 || gWl.dead || !w->configured || !st.visible) { return; }
        // Without a viewport the buffer must be a multiple of the buffer
        // scale: skip a frame drawn for the old size around a rescale.
        if (w->viewport == (void*)0 && w->intScale > 1 && ((bw % w->intScale) != 0 || (bh % w->intScale) != 0)) { return; }
        // every buffer is now stale where the frame changed
        int i = 0;
        while (i < GUI_WL_MAX_BUFFERS) {
            struct GuiWlBuffer* ob = w->bufs[i];
            if (ob != (struct GuiWlBuffer*)0 && ob->w == bw && ob->h == bh) { __wl_add_damage(ob, x, y, x + rw, y + rh); }
            i = i + 1;
        }
        struct GuiWlBuffer* b = __wl_next_buffer(w, bw, bh);
        if (b == (struct GuiWlBuffer*)0) { return; }
        __wl_add_damage(b, x, y, x + rw, y + rh);
        // Copy the stale region from the frame, RGBA -> BGRA.
        int yy = b->dy0;
        while (yy < b->dy1) {
            const unsigned char* src = (const unsigned char*)px + ((unsigned long)yy * (unsigned long)bw + (unsigned long)b->dx0) * 4UL;
            unsigned char* dst = b->data + ((unsigned long)yy * (unsigned long)bw + (unsigned long)b->dx0) * 4UL;
            int n = b->dx1 - b->dx0;
            int xx = 0;
            while (xx < n) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = src[3];
                src = src + 4;
                dst = dst + 4;
                xx = xx + 1;
            }
            yy = yy + 1;
        }
        if (w->pending == b) {
            if (x < w->pdx0) { w->pdx0 = x; }
            if (y < w->pdy0) { w->pdy0 = y; }
            if (x + rw > w->pdx1) { w->pdx1 = x + rw; }
            if (y + rh > w->pdy1) { w->pdy1 = y + rh; }
        } else {
            w->pdx0 = b->dx0; w->pdy0 = b->dy0; w->pdx1 = b->dx1; w->pdy1 = b->dy1;
        }
        b->dx0 = 0; b->dy0 = 0; b->dx1 = 0; b->dy1 = 0;
        w->pending = b;
        // Throttle to the compositor's frames: commit now unless the last
        // frame is still on its way (then the frame callback commits it; a
        // compositor that stopped sending them — window hidden — gets it
        // after 250 ms anyway).
        if (w->frameCb == (void*)0 || __wl_now_ms() - w->frameAt > 250.0) {
            if (w->frameCb != (void*)0) { gWlLib.proxy_destroy(w->frameCb); w->frameCb = (void*)0; }
            __wl_commit_pending(w);
        }
    }
}

static void __wl_set_size(&GuiWindowState st, int lw, int lh) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0) { return; }
        __wl_set_logical(w, lw, lh);
        if (!st.resizable) { __wl_toplevel_limits(w); }
        gWlLib.display_flush(gWl.display);
    }
}

static void __wl_set_position(&GuiWindowState st, int x, int y) { }

static void __wl_update_limits(&GuiWindowState st) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0) { return; }
        __wl_toplevel_limits(w);
        if (w->configured) { __wl_commit(w); }
        gWlLib.display_flush(gWl.display);
    }
}

static void __wl_set_visible(&GuiWindowState st, int v) {
    struct GuiWlWindow* w = __wl_win(st);
    unsafe {
        if (w == (struct GuiWlWindow*)0) { return; }
        if (v) {
            if (!w->configured) { __wl_map(w); }
        } else {
            // A null buffer unmaps the surface; showing it again starts
            // over with an initial commit and configure.
            struct GuiWlArgs a = __wl_args();
            __wl_set_p(&a, 0, (const void*)0);
            __wl_send(w->surface, 1U /* attach */, (const void*)0, 0U, &a, 0);
            __wl_commit(w);
            w->configured = 0;
            w->hasContent = 0;
            w->pending = (struct GuiWlBuffer*)0;
            // an unmapped surface gets no frame callbacks
            if (w->frameCb != (void*)0) { gWlLib.proxy_destroy(w->frameCb); w->frameCb = (void*)0; }
            int i = 0;
            while (i < GUI_WL_MAX_BUFFERS) {
                if (w->bufs[i] != (struct GuiWlBuffer*)0) {
                    w->bufs[i]->busy = 0;
                    w->bufs[i]->dx0 = 0; w->bufs[i]->dy0 = 0;
                    w->bufs[i]->dx1 = w->bufs[i]->w; w->bufs[i]->dy1 = w->bufs[i]->h;
                }
                i = i + 1;
            }
            gWlLib.display_flush(gWl.display);
        }
    }
}

// Wayland has no way for a client to raise or focus its own window
// without an activation token from another client (xdg-activation).
static void __wl_focus(&GuiWindowState st) { }

// ── cursors ─────────────────────────────────────────────────────────────

static unsigned int __wl_shape(int c) {
    if (c == GUI_CURSOR_IBEAM) { return 9U; }        // text
    if (c == GUI_CURSOR_HAND) { return 4U; }         // pointer
    if (c == GUI_CURSOR_CROSSHAIR) { return 8U; }
    if (c == GUI_CURSOR_RESIZE_EW) { return 26U; }
    if (c == GUI_CURSOR_RESIZE_NS) { return 27U; }
    if (c == GUI_CURSOR_RESIZE_NWSE) { return 29U; }
    if (c == GUI_CURSOR_RESIZE_NESW) { return 28U; }
    if (c == GUI_CURSOR_MOVE) { return 13U; }
    if (c == GUI_CURSOR_WAIT) { return 6U; }
    if (c == GUI_CURSOR_NOT_ALLOWED) { return 15U; }
    return 1U;                                       // default
}

// struct wl_cursor / wl_cursor_image (wayland-cursor.h)
struct GuiWlCursor {
    unsigned int imageCount;
    void** images;
    char* name;
};

struct GuiWlCursorImage {
    unsigned int width;
    unsigned int height;
    unsigned int hotspotX;
    unsigned int hotspotY;
    unsigned int delay;
};

static void* __wl_theme_cursor(int c) {
    unsafe {
        const char* a = "left_ptr"; const char* b = "default";
        if (c == GUI_CURSOR_IBEAM) { a = "xterm"; b = "text"; }
        else if (c == GUI_CURSOR_HAND) { a = "hand2"; b = "pointer"; }
        else if (c == GUI_CURSOR_CROSSHAIR) { a = "crosshair"; b = "cross"; }
        else if (c == GUI_CURSOR_RESIZE_EW) { a = "sb_h_double_arrow"; b = "ew-resize"; }
        else if (c == GUI_CURSOR_RESIZE_NS) { a = "sb_v_double_arrow"; b = "ns-resize"; }
        else if (c == GUI_CURSOR_RESIZE_NWSE) { a = "bottom_right_corner"; b = "nwse-resize"; }
        else if (c == GUI_CURSOR_RESIZE_NESW) { a = "bottom_left_corner"; b = "nesw-resize"; }
        else if (c == GUI_CURSOR_MOVE) { a = "fleur"; b = "move"; }
        else if (c == GUI_CURSOR_WAIT) { a = "watch"; b = "wait"; }
        else if (c == GUI_CURSOR_NOT_ALLOWED) { a = "crossed_circle"; b = "not-allowed"; }
        void* cur = gWlLib.cursor_theme_get_cursor(gWl.cursorTheme, a);
        if (cur == (void*)0) { cur = gWlLib.cursor_theme_get_cursor(gWl.cursorTheme, b); }
        if (cur == (void*)0) { cur = gWlLib.cursor_theme_get_cursor(gWl.cursorTheme, "left_ptr"); }
        return cur;
    }
}

// Shows cursor 'c' on the pointer (which is in a window of scale 'scale').
static void __wl_apply_cursor(int c, double scale) {
    unsafe {
        if (gWl.pointer == (void*)0 || gWl.ptrFocus == 0ULL) { return; }
        gWl.cursorShown = c;
        if (c == GUI_CURSOR_HIDDEN) {
            struct GuiWlArgs h = __wl_args();
            __wl_set_u(&h, 0, gWl.ptrSerial);
            __wl_send(gWl.pointer, 0U /* set_cursor */, (const void*)0, 0U, &h, 0);
            return;
        }
        if (gWl.shapeDevice != (void*)0) {
            struct GuiWlArgs s = __wl_args();
            __wl_set_u(&s, 0, gWl.ptrSerial);
            __wl_set_u(&s, 1, __wl_shape(c));
            __wl_send(gWl.shapeDevice, 1U /* set_shape */, (const void*)0, 0U, &s, 0);
            return;
        }
        if (!gWlLib.cursor) { return; }   // the compositor keeps whatever it showed
        int bs = (int)(scale + 0.999);
        if (bs < 1) { bs = 1; }
        int size = 24;
        const char* env = getenv("XCURSOR_SIZE");
        if (env != (const char*)0) {
            int v = (int)atof(env);
            if (v >= 8 && v <= 256) { size = v; }
        }
        if (gWl.cursorTheme == (void*)0 || gWl.cursorThemeSize != size * bs) {
            if (gWl.cursorTheme != (void*)0) { gWlLib.cursor_theme_destroy(gWl.cursorTheme); }
            gWl.cursorTheme = gWlLib.cursor_theme_load(getenv("XCURSOR_THEME"), size * bs, gWl.shm);
            gWl.cursorThemeSize = size * bs;
        }
        if (gWl.cursorTheme == (void*)0) { return; }
        struct GuiWlCursor* cur = (struct GuiWlCursor*)__wl_theme_cursor(c);
        if (cur == (struct GuiWlCursor*)0 || cur->imageCount == 0U) { return; }
        struct GuiWlCursorImage* img = (struct GuiWlCursorImage*)cur->images[0];
        void* buffer = gWlLib.cursor_image_get_buffer((void*)img);
        if (buffer == (void*)0) { return; }
        if (gWl.cursorSurface == (void*)0) {
            struct GuiWlArgs a = __wl_args();
            gWl.cursorSurface = __wl_new(gWl.compositor, 0U, gWlLib.iSurface, &a);
        }
        void* cs = gWl.cursorSurface;
        struct GuiWlArgs at = __wl_args();
        __wl_set_p(&at, 0, buffer);
        __wl_send(cs, 1U /* attach */, (const void*)0, 0U, &at, 0);
        if (gWl.compositorVer >= 3) {
            struct GuiWlArgs sc = __wl_args();
            __wl_set_i(&sc, 0, bs);
            __wl_send(cs, 8U /* set_buffer_scale */, (const void*)0, 0U, &sc, 0);
        }
        struct GuiWlArgs dm = __wl_args();
        __wl_set_i(&dm, 2, 0x7FFFFFFF);
        __wl_set_i(&dm, 3, 0x7FFFFFFF);
        __wl_send(cs, 2U /* damage */, (const void*)0, 0U, &dm, 0);
        __wl_req0(cs, 6U /* commit */);
        struct GuiWlArgs set = __wl_args();
        __wl_set_u(&set, 0, gWl.ptrSerial);
        __wl_set_p(&set, 1, cs);
        __wl_set_i(&set, 2, (int)img->hotspotX / bs);
        __wl_set_i(&set, 3, (int)img->hotspotY / bs);
        __wl_send(gWl.pointer, 0U /* set_cursor */, (const void*)0, 0U, &set, 0);
    }
}

static void __wl_set_cursor(&GuiWindowState st, int c) {
    unsafe {
        if (gWl.ptrFocus == st.key) {
            __wl_apply_cursor(c, st.scale);
            gWlLib.display_flush(gWl.display);
        }
    }
}

// ── keys ────────────────────────────────────────────────────────────────

// X11/xkb keysyms to GuiKey (the same table as the X11 backend's).
static int __wl_keysym(unsigned int ks) {
    if (ks == 0xFF51U || ks == 0xFF96U) { return GUI_KEY_LEFT; }
    if (ks == 0xFF53U || ks == 0xFF98U) { return GUI_KEY_RIGHT; }
    if (ks == 0xFF52U || ks == 0xFF97U) { return GUI_KEY_UP; }
    if (ks == 0xFF54U || ks == 0xFF99U) { return GUI_KEY_DOWN; }
    if (ks == 0xFF0DU || ks == 0xFF8DU) { return GUI_KEY_ENTER; }
    if (ks == 0xFF1BU) { return GUI_KEY_ESCAPE; }
    if (ks == 0xFF09U || ks == 0xFE20U) { return GUI_KEY_TAB; }
    if (ks == 0xFF08U) { return GUI_KEY_BACKSPACE; }
    if (ks == 0x20U)   { return GUI_KEY_SPACE; }
    if (ks == 0xFFFFU || ks == 0xFF9FU) { return GUI_KEY_DELETE; }
    if (ks == 0xFF50U || ks == 0xFF95U) { return GUI_KEY_HOME; }
    if (ks == 0xFF57U || ks == 0xFF9CU) { return GUI_KEY_END; }
    if (ks == 0xFF55U || ks == 0xFF9AU) { return GUI_KEY_PAGE_UP; }
    if (ks == 0xFF56U || ks == 0xFF9BU) { return GUI_KEY_PAGE_DOWN; }
    if (ks == 0xFF63U || ks == 0xFF9EU) { return GUI_KEY_INSERT; }
    if (ks == 0xFF67U) { return GUI_KEY_MENU; }
    if (ks >= 0xFFBEU && ks <= 0xFFC9U) { return GUI_KEY_F1 + (int)(ks - 0xFFBEU); }
    if (ks == 0xFFE1U || ks == 0xFFE2U) { return GUI_KEY_SHIFT; }
    if (ks == 0xFFE3U || ks == 0xFFE4U) { return GUI_KEY_CONTROL; }
    if (ks == 0xFFE9U || ks == 0xFFEAU) { return GUI_KEY_ALT; }
    if (ks == 0xFFE7U || ks == 0xFFE8U || ks == 0xFFEBU || ks == 0xFFECU) { return GUI_KEY_META; }
    if (ks >= 0x61U && ks <= 0x7AU) { return (int)(ks - 0x20U); }
    if (ks >= 0x41U && ks <= 0x5AU) { return (int)ks; }
    if (ks >= 0x30U && ks <= 0x39U) { return (int)ks; }
    return GUI_KEY_UNKNOWN;
}

// Without libxkbcommon: Linux evdev key codes as a US keyboard, as
// (keysym unshifted, keysym shifted).
static unsigned int __wl_evdev_sym(unsigned int code, int shift) {
    const char* row1 = "1234567890-=";   const char* row1s = "!@#$%^&*()_+";
    const char* row2 = "qwertyuiop[]";   const char* row2s = "QWERTYUIOP{}";
    const char* row3 = "asdfghjkl;'`";   const char* row3s = "ASDFGHJKL:\"~";
    const char* row4 = "\\zxcvbnm,./";   const char* row4s = "|ZXCVBNM<>?";
    unsafe {
        if (code >= 2U && code <= 13U) { return (unsigned int)(unsigned char)(shift ? row1s : row1)[code - 2U]; }
        if (code >= 16U && code <= 27U) { return (unsigned int)(unsigned char)(shift ? row2s : row2)[code - 16U]; }
        if (code >= 30U && code <= 41U) { return (unsigned int)(unsigned char)(shift ? row3s : row3)[code - 30U]; }
        if (code >= 43U && code <= 53U) { return (unsigned int)(unsigned char)(shift ? row4s : row4)[code - 43U]; }
    }
    if (code == 1U) { return 0xFF1BU; }
    if (code == 14U) { return 0xFF08U; }
    if (code == 15U) { return 0xFF09U; }
    if (code == 28U || code == 96U) { return 0xFF0DU; }
    if (code == 29U) { return 0xFFE3U; }
    if (code == 97U) { return 0xFFE4U; }
    if (code == 42U) { return 0xFFE1U; }
    if (code == 54U) { return 0xFFE2U; }
    if (code == 56U) { return 0xFFE9U; }
    if (code == 100U) { return 0xFFEAU; }
    if (code == 57U) { return 0x20U; }
    if (code >= 59U && code <= 68U) { return 0xFFBEU + (code - 59U); }
    if (code == 87U) { return 0xFFC8U; }
    if (code == 88U) { return 0xFFC9U; }
    if (code == 102U) { return 0xFF50U; }
    if (code == 103U) { return 0xFF52U; }
    if (code == 104U) { return 0xFF55U; }
    if (code == 105U) { return 0xFF51U; }
    if (code == 106U) { return 0xFF53U; }
    if (code == 107U) { return 0xFF57U; }
    if (code == 108U) { return 0xFF54U; }
    if (code == 109U) { return 0xFF56U; }
    if (code == 110U) { return 0xFF63U; }
    if (code == 111U) { return 0xFFFFU; }
    if (code == 125U) { return 0xFFEBU; }
    if (code == 126U) { return 0xFFECU; }
    if (code == 127U) { return 0xFF67U; }
    return 0U;
}

static unsigned int __wl_keysym_char(unsigned int ks) {
    if (ks >= 0x20U && ks <= 0x7EU) { return ks; }
    if (ks >= 0xA0U && ks <= 0xFFU) { return ks; }
    if (ks >= 0x01000100U && ks <= 0x0110FFFFU) { return ks - 0x01000000U; }
    return 0U;
}

static void __wl_free_keymap() {
    unsafe {
        if (gWl.xkbState != (void*)0) { gWlLib.xkb_state_unref(gWl.xkbState); gWl.xkbState = (void*)0; }
        if (gWl.keymap != (void*)0) { gWlLib.xkb_keymap_unref(gWl.keymap); gWl.keymap = (void*)0; }
    }
}

static void __wl_keymap(unsigned int format, int fd, unsigned int size) {
    unsafe {
        if (format != 1U /* XKB_V1 */ || !gWlLib.xkb) { close(fd); return; }
        void* map = mmap((void*)0, (unsigned long)size, 1 /* PROT_READ */, 2 /* MAP_PRIVATE */, fd, 0L);
        close(fd);
        if (map == (void*)0 || (long)map == -1L) { return; }
        if (gWl.xkbCtx == (void*)0) {
            gWl.xkbCtx = gWlLib.xkb_context_new(0);
            if (gWl.xkbCtx != (void*)0 && gWlLib.xkb_compose_table_new_from_locale != (XkbComposeTableNew_)0) {
                const char* loc = getenv("LC_ALL");
                if (loc == (const char*)0 || loc[0] == '\0') { loc = getenv("LC_CTYPE"); }
                if (loc == (const char*)0 || loc[0] == '\0') { loc = getenv("LANG"); }
                if (loc == (const char*)0 || loc[0] == '\0') { loc = "C"; }
                void* table = gWlLib.xkb_compose_table_new_from_locale(gWl.xkbCtx, loc, 0);
                if (table != (void*)0) { gWl.compose = gWlLib.xkb_compose_state_new(table, 0); }
            }
        }
        if (gWl.xkbCtx == (void*)0) { munmap(map, (unsigned long)size); return; }
        void* km = gWlLib.xkb_keymap_new_from_string(gWl.xkbCtx, (const char*)map, 1, 0);
        munmap(map, (unsigned long)size);
        if (km == (void*)0) { return; }
        __wl_free_keymap();
        gWl.keymap = km;
        gWl.xkbState = gWlLib.xkb_state_new(km);
    }
}

static int __wl_mods_from_xkb() {
    unsafe {
        int m = 0;
        void* s = gWl.xkbState;
        if (gWlLib.xkb_state_mod_name_is_active(s, "Shift", 8) > 0) { m = m | GUI_MOD_SHIFT; }
        if (gWlLib.xkb_state_mod_name_is_active(s, "Control", 8) > 0) { m = m | GUI_MOD_CTRL; }
        if (gWlLib.xkb_state_mod_name_is_active(s, "Mod1", 8) > 0) { m = m | GUI_MOD_ALT; }
        if (gWlLib.xkb_state_mod_name_is_active(s, "Mod4", 8) > 0) { m = m | GUI_MOD_META; }
        return m;
    }
}

static void __wl_key(unsigned int serial, unsigned int key, unsigned int state) {
    unsafe {
        gWl.lastSerial = serial;
        struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.kbFocus);
        int down = state == 1U ? 1 : 0;   // 2 (repeated, wl_keyboard v10) never reaches v8 clients
        if (!down && gWl.repeatKey == key + 8U) { gWl.repeatKey = 0U; }
        if (st == (struct GuiWindowState*)0) { return; }
        unsigned int code = key + 8U;
        struct GuiEvent ev = GuiEvent::new(down ? GUI_EVENT_KEY_DOWN : GUI_EVENT_KEY_UP);
        ev.modifiers = gWl.mods;
        ev.x = gWl.ptrX * st->scale;
        ev.y = gWl.ptrY * st->scale;
        unsigned int cp = 0U;
        if (gWl.xkbState != (void*)0) {
            unsigned int layout = gWlLib.xkb_state_key_get_layout(gWl.xkbState, code);
            const unsigned int* syms = (const unsigned int*)0;
            int n = gWlLib.xkb_keymap_key_get_syms_by_level(gWl.keymap, code, layout, 0U, &syms);
            unsigned int sym = gWlLib.xkb_state_key_get_one_sym(gWl.xkbState, code);
            ev.keycode = n > 0 ? __wl_keysym(syms[0]) : GUI_KEY_UNKNOWN;
            if (ev.keycode == GUI_KEY_UNKNOWN) { ev.keycode = __wl_keysym(sym); }
            if (down) {
                int composed = 0;
                if (gWl.compose != (void*)0) {
                    gWlLib.xkb_compose_state_feed(gWl.compose, sym);
                    int cs = gWlLib.xkb_compose_state_get_status(gWl.compose);
                    if (cs == 1 /* COMPOSING */) { composed = 1; cp = 0U; }
                    else if (cs == 2 /* COMPOSED */) {
                        cp = __wl_keysym_char(gWlLib.xkb_compose_state_get_one_sym(gWl.compose));
                        gWlLib.xkb_compose_state_reset(gWl.compose);
                        composed = 1;
                    } else if (cs == 3 /* CANCELLED */) {
                        gWlLib.xkb_compose_state_reset(gWl.compose);
                        composed = 1;
                    }
                }
                if (!composed) { cp = gWlLib.xkb_state_key_get_utf32(gWl.xkbState, code); }
            }
        } else {
            unsigned int sym0 = __wl_evdev_sym(key, 0);
            ev.keycode = __wl_keysym(sym0);
            if (down) { cp = __wl_keysym_char(__wl_evdev_sym(key, (gWl.mods & GUI_MOD_SHIFT) != 0 ? 1 : 0)); }
        }
        if (cp < 32U || cp == 127U || (gWl.mods & (GUI_MOD_CTRL | GUI_MOD_META)) != 0) { cp = 0U; }
        ev.codepoint = cp;
        gui_backend_post(st, ev);
        if (down) {
            int repeats = 1;
            if (gWl.keymap != (void*)0) { repeats = gWlLib.xkb_keymap_key_repeats(gWl.keymap, code); }
            if (ev.keycode == GUI_KEY_SHIFT || ev.keycode == GUI_KEY_CONTROL || ev.keycode == GUI_KEY_ALT || ev.keycode == GUI_KEY_META) { repeats = 0; }
            if (repeats && gWl.repeatRate > 0) {
                gWl.repeatKey = code;
                gWl.repeatEv = ev;
                gWl.repeatNext = __wl_now_ms() + (double)gWl.repeatDelay;
            }
        }
    }
}

static void __wl_key_repeat() {
    unsafe {
        if (gWl.repeatKey == 0U || gWl.repeatRate <= 0) { return; }
        double now = __wl_now_ms();
        if (now < gWl.repeatNext) { return; }
        struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.kbFocus);
        if (st == (struct GuiWindowState*)0) { gWl.repeatKey = 0U; return; }
        int n = 0;
        while (now >= gWl.repeatNext && n < 8) {
            struct GuiEvent ev = gWl.repeatEv;
            ev.timeMs = 0.0;
            gui_backend_post(st, ev);
            gWl.repeatNext = gWl.repeatNext + 1000.0 / (double)gWl.repeatRate;
            n = n + 1;
        }
        if (now >= gWl.repeatNext) { gWl.repeatNext = now + 1000.0 / (double)gWl.repeatRate; }
    }
}

static int __wl_keyboard_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 0U) {          // keymap(format, fd, size)
            __wl_keymap(__wl_arg_u(args, 0), __wl_arg_i(args, 1), __wl_arg_u(args, 2));
        } else if (opcode == 1U) {   // enter(serial, surface, keys)
            gWl.lastSerial = __wl_arg_u(args, 0);
            void* surface = __wl_arg_p(args, 1);
            gWl.kbFocus = (unsigned long long)(unsigned long)surface;
            struct GuiWindowState* st = __wl_find(surface);
            if (st != (struct GuiWindowState*)0 && !st->focused) {
                st->focused = 1;
                gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_IN));
            }
        } else if (opcode == 2U) {   // leave(serial, surface)
            void* surface = __wl_arg_p(args, 1);
            struct GuiWindowState* st = __wl_find(surface);
            if (gWl.kbFocus == (unsigned long long)(unsigned long)surface) { gWl.kbFocus = 0ULL; }
            gWl.repeatKey = 0U;
            if (st != (struct GuiWindowState*)0 && st->focused) {
                st->focused = 0;
                gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_OUT));
            }
        } else if (opcode == 3U) {   // key(serial, time, key, state)
            __wl_key(__wl_arg_u(args, 0), __wl_arg_u(args, 2), __wl_arg_u(args, 3));
        } else if (opcode == 4U) {   // modifiers(serial, depressed, latched, locked, group)
            unsigned int dep = __wl_arg_u(args, 1);
            unsigned int lat = __wl_arg_u(args, 2);
            unsigned int lck = __wl_arg_u(args, 3);
            unsigned int grp = __wl_arg_u(args, 4);
            if (gWl.xkbState != (void*)0) {
                gWlLib.xkb_state_update_mask(gWl.xkbState, dep, lat, lck, 0U, 0U, grp);
                gWl.mods = __wl_mods_from_xkb();
            } else {
                unsigned int m = dep | lat;
                int mods = 0;
                if ((m & 1U) != 0U) { mods = mods | GUI_MOD_SHIFT; }
                if ((m & 4U) != 0U) { mods = mods | GUI_MOD_CTRL; }
                if ((m & 8U) != 0U) { mods = mods | GUI_MOD_ALT; }
                if ((m & 64U) != 0U) { mods = mods | GUI_MOD_META; }
                gWl.mods = mods;
            }
            if (gWl.repeatKey != 0U) { gWl.repeatEv.modifiers = gWl.mods; }
        } else if (opcode == 5U) {   // repeat_info(rate, delay)
            gWl.repeatRate = __wl_arg_i(args, 0);
            gWl.repeatDelay = __wl_arg_i(args, 1);
        }
    }
    return 0;
}

// ── pointer ─────────────────────────────────────────────────────────────

static void __wl_flush_axis() {
    unsafe {
        if (!gWl.axisHave) { return; }
        struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.ptrFocus);
        if (st != (struct GuiWindowState*)0) {
            struct GuiEvent ev = GuiEvent::new(GUI_EVENT_SCROLL);
            ev.x = gWl.ptrX * st->scale;
            ev.y = gWl.ptrY * st->scale;
            ev.modifiers = gWl.mods;
            if (gWl.axisHaveDisc) {
                ev.scrollDx = (double)gWl.axisDiscX / 120.0;
                ev.scrollDy = -(double)gWl.axisDiscY / 120.0;
            } else {
                // smooth scrolling (touchpads): ~10 surface units per line
                ev.scrollDx = gWl.axisDx / 10.0;
                ev.scrollDy = -gWl.axisDy / 10.0;
            }
            if (ev.scrollDx != 0.0 || ev.scrollDy != 0.0) { gui_backend_post(st, ev); }
        }
        gWl.axisHave = 0; gWl.axisHaveDisc = 0;
        gWl.axisDx = 0.0; gWl.axisDy = 0.0;
        gWl.axisDiscX = 0; gWl.axisDiscY = 0;
    }
}

static int __wl_pointer_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 0U) {          // enter(serial, surface, x, y)
            gWl.ptrSerial = __wl_arg_u(args, 0);
            void* surface = __wl_arg_p(args, 1);
            gWl.ptrFocus = (unsigned long long)(unsigned long)surface;
            gWl.ptrX = __wl_arg_fixed(args, 2);
            gWl.ptrY = __wl_arg_fixed(args, 3);
            struct GuiWindowState* st = __wl_find(surface);
            if (st != (struct GuiWindowState*)0) {
                __wl_apply_cursor(st->cursor, st->scale);
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOUSE_ENTER);
                ev.x = gWl.ptrX * st->scale;
                ev.y = gWl.ptrY * st->scale;
                gui_backend_post(st, ev);
            }
        } else if (opcode == 1U) {   // leave(serial, surface)
            void* surface = __wl_arg_p(args, 1);
            struct GuiWindowState* st = __wl_find(surface);
            if (gWl.ptrFocus == (unsigned long long)(unsigned long)surface) { gWl.ptrFocus = 0ULL; }
            if (st != (struct GuiWindowState*)0) {
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOUSE_LEAVE);
                ev.x = gWl.ptrX * st->scale;
                ev.y = gWl.ptrY * st->scale;
                gui_backend_post(st, ev);
            }
        } else if (opcode == 2U) {   // motion(time, x, y)
            gWl.ptrX = __wl_arg_fixed(args, 1);
            gWl.ptrY = __wl_arg_fixed(args, 2);
            struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.ptrFocus);
            if (st != (struct GuiWindowState*)0) {
                struct GuiEvent ev = GuiEvent::new(GUI_EVENT_MOUSE_MOVE);
                ev.x = gWl.ptrX * st->scale;
                ev.y = gWl.ptrY * st->scale;
                ev.modifiers = gWl.mods;
                gui_backend_post(st, ev);
            }
        } else if (opcode == 3U) {   // button(serial, time, button, state)
            gWl.lastSerial = __wl_arg_u(args, 0);
            unsigned int b = __wl_arg_u(args, 2);
            unsigned int state = __wl_arg_u(args, 3);
            struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.ptrFocus);
            if (st != (struct GuiWindowState*)0) {
                struct GuiEvent ev = GuiEvent::new(state == 1U ? GUI_EVENT_MOUSE_DOWN : GUI_EVENT_MOUSE_UP);
                ev.x = gWl.ptrX * st->scale;
                ev.y = gWl.ptrY * st->scale;
                ev.modifiers = gWl.mods;
                if (b == 0x110U) { ev.button = GUI_BUTTON_LEFT; }
                else if (b == 0x111U) { ev.button = GUI_BUTTON_RIGHT; }
                else { ev.button = GUI_BUTTON_MIDDLE; }
                gui_backend_post(st, ev);
            }
        } else if (opcode == 4U) {   // axis(time, axis, value)
            unsigned int axis = __wl_arg_u(args, 1);
            double v = __wl_arg_fixed(args, 2);
            if (axis == 0U) { gWl.axisDy = gWl.axisDy + v; } else { gWl.axisDx = gWl.axisDx + v; }
            gWl.axisHave = 1;
            if (gWl.pointerVer < 5) { __wl_flush_axis(); }
        } else if (opcode == 5U) {   // frame
            __wl_flush_axis();
        } else if (opcode == 8U || opcode == 9U) {   // axis_discrete(axis, steps) / axis_value120(axis, v120)
            unsigned int axis = __wl_arg_u(args, 0);
            int v = __wl_arg_i(args, 1);
            if (opcode == 8U) {
                if (gWl.pointerVer >= 8) { return 0; }   // value120 carries it
                v = v * 120;
            }
            if (axis == 0U) { gWl.axisDiscY = gWl.axisDiscY + v; } else { gWl.axisDiscX = gWl.axisDiscX + v; }
            gWl.axisHaveDisc = 1;
            gWl.axisHave = 1;
        }
    }
    return 0;
}

// ── clipboard ───────────────────────────────────────────────────────────

static int __wl_offer_index(void* proxy) {
    unsafe {
        int i = 0;
        while (i < GUI_WL_MAX_OFFERS) {
            if (gWl.offers[i].proxy == proxy && proxy != (void*)0) { return i; }
            i = i + 1;
        }
    }
    return -1;
}

static void __wl_offer_drop(int i) {
    unsafe {
        if (i < 0 || gWl.offers[i].proxy == (void*)0) { return; }
        __wl_destroy_req(gWl.offers[i].proxy, 2U /* destroy */);
        memset((void*)&gWl.offers[i], 0, sizeof(struct GuiWlOffer));
        if (gWl.selection == i) { gWl.selection = -1; }
    }
}

static int __wl_offer_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        int i = __wl_offer_index(target);
        if (i < 0) { return 0; }
        if (opcode == 0U) {   // offer(mime)
            const char* m = (const char*)__wl_arg_p(args, 0);
            if (__wl_streq(m, "text/plain;charset=utf-8")) { gWl.offers[i].utf8 = 1; }
            else if (__wl_streq(m, "text/plain")) { gWl.offers[i].plain = 1; }
            else if (__wl_streq(m, "UTF8_STRING")) { gWl.offers[i].x11utf8 = 1; }
            else if (__wl_streq(m, "STRING") || __wl_streq(m, "TEXT")) { gWl.offers[i].string = 1; }
        }
    }
    return 0;
}

static int __wl_data_device_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 0U) {          // data_offer(new id)
            void* offer = __wl_arg_p(args, 0);
            if (offer == (void*)0) { return 0; }
            int slot = -1;
            int i = 0;
            while (i < GUI_WL_MAX_OFFERS) {
                if (gWl.offers[i].proxy == (void*)0) { slot = i; break; }
                i = i + 1;
            }
            if (slot < 0) {
                // full (a drag-and-drop peer offering many): recycle a non-selection slot
                slot = gWl.selection == 0 ? 1 : 0;
                __wl_offer_drop(slot);
            }
            memset((void*)&gWl.offers[slot], 0, sizeof(struct GuiWlOffer));
            gWl.offers[slot].proxy = offer;
            gWlLib.add_dispatcher(offer, (void*)__wl_offer_dispatch, (const void*)0, (void*)0);
        } else if (opcode == 5U) {   // selection(offer or null)
            void* offer = __wl_arg_p(args, 0);
            int keep = __wl_offer_index(offer);
            int i = 0;
            while (i < GUI_WL_MAX_OFFERS) {
                if (i != keep) { __wl_offer_drop(i); }
                i = i + 1;
            }
            gWl.selection = keep;
        } else if (opcode == 2U || opcode == 4U) {   // drag leave / drop: offers for DnD, ignored
            int i = 0;
            while (i < GUI_WL_MAX_OFFERS) {
                if (i != gWl.selection) { __wl_offer_drop(i); }
                i = i + 1;
            }
        }
    }
    return 0;
}

static int __wl_source_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 1U) {          // send(mime, fd)
            int fd = __wl_arg_i(args, 1);
            unsigned long off = 0UL;
            while (off < gWl.clip.len) {
                long n = write(fd, (const void*)((unsigned long)gWl.clip.data + off), gWl.clip.len - off);
                if (n <= 0L) { break; }
                off = off + (unsigned long)n;
            }
            close(fd);
        } else if (opcode == 2U) {   // cancelled: someone else owns the clipboard now
            if (gWl.source == target) { gWl.source = (void*)0; }
            __wl_destroy_req(target, 1U /* destroy */);
        }
    }
    return 0;
}

static void __wl_ensure_data_device() {
    unsafe {
        if (gWl.dataDevice != (void*)0 || gWl.ddm == (void*)0 || gWl.seat == (void*)0) { return; }
        struct GuiWlArgs a = __wl_args();
        __wl_set_p(&a, 1, gWl.seat);
        gWl.dataDevice = __wl_new(gWl.ddm, 1U /* get_data_device */, gWlLib.iDataDevice, &a);
        if (gWl.dataDevice != (void*)0) {
            gWlLib.add_dispatcher(gWl.dataDevice, (void*)__wl_data_device_dispatch, (const void*)0, (void*)0);
        }
    }
}

static int __wl_clip_set(const &char utf8, unsigned long n) {
    unsafe {
        if (gWl.dataDevice == (void*)0 || gWl.dead) { return 0; }
        gWl.clip.free();
        gWl.clip = String::from_n_(utf8, n);
        struct GuiWlArgs a = __wl_args();
        void* src = __wl_new(gWl.ddm, 0U /* create_data_source */, gWlLib.iDataSource, &a);
        if (src == (void*)0) { return 0; }
        gWlLib.add_dispatcher(src, (void*)__wl_source_dispatch, (const void*)0, (void*)0);
        const char* mimes[5];
        mimes[0] = "text/plain;charset=utf-8"; mimes[1] = "text/plain"; mimes[2] = "UTF8_STRING";
        mimes[3] = "STRING"; mimes[4] = "TEXT";
        int i = 0;
        while (i < 5) {
            struct GuiWlArgs m = __wl_args();
            __wl_set_p(&m, 0, (const void*)mimes[i]);
            __wl_send(src, 0U /* offer */, (const void*)0, 0U, &m, 0);
            i = i + 1;
        }
        void* old = gWl.source;
        gWl.source = src;
        struct GuiWlArgs s = __wl_args();
        __wl_set_p(&s, 0, src);
        __wl_set_u(&s, 1, gWl.lastSerial);
        __wl_send(gWl.dataDevice, 1U /* set_selection */, (const void*)0, 0U, &s, 0);
        if (old != (void*)0) { __wl_destroy_req(old, 1U); }
        // The compositor cancels the source at once if it refuses (no
        // focus, stale serial).
        if (gWlLib.display_roundtrip(gWl.display) < 0) { gWl.dead = 1; return 0; }
        return gWl.source == src ? 1 : 0;
    }
}

struct GuiWlPollFd {
    int fd;
    short events;
    short revents;
};

static int __wl_clip_get(&String out) {
    out.clear();
    unsafe {
        if (gWl.dead) { return 0; }
        gWlLib.display_roundtrip(gWl.display);
        if (gWl.selection < 0) { return 0; }
        // Is the selection ours? (Our own source shows up as an offer too.)
        if (gWl.source != (void*)0) {
            out.push_n_((const char*)gWl.clip.data, gWl.clip.len);
            return 1;
        }
        struct GuiWlOffer o = gWl.offers[gWl.selection];
        const char* mime = (const char*)0;
        if (o.utf8) { mime = "text/plain;charset=utf-8"; }
        else if (o.x11utf8) { mime = "UTF8_STRING"; }
        else if (o.plain) { mime = "text/plain"; }
        else if (o.string) { mime = "STRING"; }
        if (mime == (const char*)0) { return 0; }
        int fds[2];
        if (pipe(&fds[0]) != 0) { return 0; }
        struct GuiWlArgs a = __wl_args();
        __wl_set_p(&a, 0, (const void*)mime);
        __wl_set_i(&a, 1, fds[1]);
        __wl_send(o.proxy, 1U /* receive */, (const void*)0, 0U, &a, 0);
        gWlLib.display_flush(gWl.display);
        close(fds[1]);
        double start = __wl_now_ms();
        char buf[4096];
        int ok = 0;
        while (1) {
            double left = 1000.0 - (__wl_now_ms() - start);
            if (left <= 0.0) { break; }
            struct GuiWlPollFd p;
            p.fd = fds[0]; p.events = (short)1; p.revents = (short)0;
            int r = poll((void*)&p, 1UL, (int)left);
            if (r <= 0) { break; }
            long n = read(fds[0], (void*)&buf[0], 4096UL);
            if (n < 0L) { break; }
            if (n == 0L) { ok = 1; break; }
            out.push_n_((const char*)&buf[0], (unsigned long)n);
        }
        close(fds[0]);
        return ok;
    }
}

// ── registry, seat, outputs ─────────────────────────────────────────────

static int __wl_output_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 3U) {   // scale(factor)
            int i = 0;
            while (i < gWl.outputCount) {
                if (gWl.outputs[i].proxy == target) { gWl.outputs[i].scale = __wl_arg_i(args, 0); }
                i = i + 1;
            }
        }
    }
    return 0;
}

static void __wl_seat_caps(unsigned int caps) {
    unsafe {
        int wantPointer = (caps & 1U) != 0U ? 1 : 0;
        int wantKeyboard = (caps & 2U) != 0U ? 1 : 0;
        if (wantPointer && gWl.pointer == (void*)0) {
            struct GuiWlArgs a = __wl_args();
            gWl.pointer = __wl_new(gWl.seat, 0U /* get_pointer */, gWlLib.iPointer, &a);
            gWl.pointerVer = gWl.seatVer;
            gWlLib.add_dispatcher(gWl.pointer, (void*)__wl_pointer_dispatch, (const void*)0, (void*)0);
            if (gWl.shapeManager != (void*)0) {
                struct GuiWlArgs s = __wl_args();
                __wl_set_p(&s, 1, gWl.pointer);
                gWl.shapeDevice = __wl_new(gWl.shapeManager, 1U /* get_pointer */, (const void*)&gWlShapeDevice.iface, &s);
            }
        } else if (!wantPointer && gWl.pointer != (void*)0) {
            if (gWl.shapeDevice != (void*)0) { __wl_destroy_req(gWl.shapeDevice, 0U); gWl.shapeDevice = (void*)0; }
            if (gWl.seatVer >= 3) { __wl_destroy_req(gWl.pointer, 1U /* release */); }
            else { gWlLib.proxy_destroy(gWl.pointer); }
            gWl.pointer = (void*)0;
            gWl.ptrFocus = 0ULL;
        }
        if (wantKeyboard && gWl.keyboard == (void*)0) {
            struct GuiWlArgs a = __wl_args();
            gWl.keyboard = __wl_new(gWl.seat, 1U /* get_keyboard */, gWlLib.iKeyboard, &a);
            gWl.repeatRate = 25;
            gWl.repeatDelay = 600;
            gWlLib.add_dispatcher(gWl.keyboard, (void*)__wl_keyboard_dispatch, (const void*)0, (void*)0);
        } else if (!wantKeyboard && gWl.keyboard != (void*)0) {
            if (gWl.seatVer >= 3) { __wl_destroy_req(gWl.keyboard, 0U /* release */); }
            else { gWlLib.proxy_destroy(gWl.keyboard); }
            gWl.keyboard = (void*)0;
            gWl.repeatKey = 0U;
            // keyboard focus goes with the keyboard
            struct GuiWindowState* st = __wl_find((void*)(unsigned long)gWl.kbFocus);
            gWl.kbFocus = 0ULL;
            if (st != (struct GuiWindowState*)0 && st->focused) {
                st->focused = 0;
                gui_backend_post(st, GuiEvent::new(GUI_EVENT_FOCUS_OUT));
            }
        }
    }
}

static int __wl_seat_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    if (opcode == 0U) { __wl_seat_caps(__wl_arg_u(args, 0)); }
    return 0;
}

static int __wl_wm_base_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    if (opcode == 0U) {   // ping(serial)
        struct GuiWlArgs a = __wl_args();
        __wl_set_u(&a, 0, __wl_arg_u(args, 0));
        __wl_send(target, 3U /* pong */, (const void*)0, 0U, &a, 0);
    }
    return 0;
}

static unsigned int __wl_minu(unsigned int a, unsigned int b) { return a < b ? a : b; }

static void* __wl_bind(unsigned int name, const void* iface, unsigned int version) {
    unsafe {
        struct GuiWlArgs a = __wl_args();
        __wl_set_u(&a, 0, name);
        __wl_set_p(&a, 1, (const void*)((const struct GuiWlInterface*)iface)->name);
        __wl_set_u(&a, 2, version);
        return __wl_send(gWl.registry, 0U /* bind */, iface, version, &a, 0);
    }
}

static int __wl_registry_dispatch(const void* data, void* target, unsigned int opcode, const void* msg, void* args) {
    unsafe {
        if (opcode == 0U) {          // global(name, interface, version)
            unsigned int name = __wl_arg_u(args, 0);
            const char* iface = (const char*)__wl_arg_p(args, 1);
            unsigned int ver = __wl_arg_u(args, 2);
            if (__wl_streq(iface, "wl_compositor") && gWl.compositor == (void*)0) {
                unsigned int v = __wl_minu(__wl_minu(ver, 6U), (unsigned int)__wl_iface_version(gWlLib.iSurface));
                gWl.compositor = __wl_bind(name, gWlLib.iCompositor, v);
                gWl.compositorVer = (int)v;
            } else if (__wl_streq(iface, "wl_shm") && gWl.shm == (void*)0) {
                gWl.shm = __wl_bind(name, gWlLib.iShm, 1U);
            } else if (__wl_streq(iface, "xdg_wm_base") && gWl.wmBase == (void*)0) {
                gWl.wmBase = __wl_bind(name, (const void*)&gWlXdgWmBase.iface, __wl_minu(ver, 6U));
                gWlLib.add_dispatcher(gWl.wmBase, (void*)__wl_wm_base_dispatch, (const void*)0, (void*)0);
            } else if (__wl_streq(iface, "wl_seat") && gWl.seat == (void*)0) {
                unsigned int v = __wl_minu(__wl_minu(ver, 8U), (unsigned int)__wl_iface_version(gWlLib.iSeat));
                gWl.seat = __wl_bind(name, gWlLib.iSeat, v);
                gWl.seatName = name;
                gWl.seatVer = (int)v;
                gWlLib.add_dispatcher(gWl.seat, (void*)__wl_seat_dispatch, (const void*)0, (void*)0);
            } else if (__wl_streq(iface, "wl_output") && gWl.outputCount < GUI_WL_MAX_OUTPUTS) {
                unsigned int v = __wl_minu(__wl_minu(ver, 2U), (unsigned int)__wl_iface_version(gWlLib.iOutput));
                void* o = __wl_bind(name, gWlLib.iOutput, v);
                gWl.outputs[gWl.outputCount].proxy = o;
                gWl.outputs[gWl.outputCount].name = name;
                gWl.outputs[gWl.outputCount].scale = 1;
                gWl.outputCount = gWl.outputCount + 1;
                gWlLib.add_dispatcher(o, (void*)__wl_output_dispatch, (const void*)0, (void*)0);
            } else if (__wl_streq(iface, "wl_data_device_manager") && gWl.ddm == (void*)0) {
                unsigned int v = __wl_minu(__wl_minu(ver, 3U), (unsigned int)__wl_iface_version(gWlLib.iDataDeviceManager));
                gWl.ddm = __wl_bind(name, gWlLib.iDataDeviceManager, v);
            } else if (__wl_streq(iface, "zxdg_decoration_manager_v1") && gWl.decoManager == (void*)0) {
                gWl.decoManager = __wl_bind(name, (const void*)&gWlDecoManager.iface, 1U);
            } else if (__wl_streq(iface, "wp_viewporter") && gWl.viewporter == (void*)0) {
                gWl.viewporter = __wl_bind(name, (const void*)&gWlViewporter.iface, 1U);
            } else if (__wl_streq(iface, "wp_fractional_scale_manager_v1") && gWl.fracManager == (void*)0) {
                gWl.fracManager = __wl_bind(name, (const void*)&gWlFracManager.iface, 1U);
            } else if (__wl_streq(iface, "wp_cursor_shape_manager_v1") && gWl.shapeManager == (void*)0) {
                gWl.shapeManager = __wl_bind(name, (const void*)&gWlShapeManager.iface, 1U);
            }
        } else if (opcode == 1U) {   // global_remove(name)
            unsigned int name = __wl_arg_u(args, 0);
            int i = 0;
            while (i < gWl.outputCount) {
                if (gWl.outputs[i].name == name) {
                    gWl.outputs[i] = gWl.outputs[gWl.outputCount - 1];
                    gWl.outputCount = gWl.outputCount - 1;
                    break;
                }
                i = i + 1;
            }
        }
    }
    return 0;
}

// ── connection ──────────────────────────────────────────────────────────

static int __wl_connect() {
    if (gWlState != 0) { return gWlState > 0 ? 1 : 0; }
    gWlState = -1;
    if (!__wl_load()) { return 0; }
    unsafe {
        memset((void*)&gWl, 0, sizeof(struct GuiWl));
        gWl.selection = -1;
        gWl.wakeR = -1;
        gWl.wakeW = -1;
        void* d = gWlLib.display_connect((const char*)0);
        if (d == (void*)0) { return 0; }
        gWl.display = d;
        __wl_build_protocols();
        struct GuiWlArgs a = __wl_args();
        gWl.registry = __wl_send(d, 1U /* get_registry */, gWlLib.iRegistry, 1U, &a, 0);
        gWlLib.add_dispatcher(gWl.registry, (void*)__wl_registry_dispatch, (const void*)0, (void*)0);
        // globals, then the events binding them caused (seat caps, output scales)
        if (gWlLib.display_roundtrip(d) < 0 || gWlLib.display_roundtrip(d) < 0) {
            gWlLib.display_disconnect(d);
            gWl.display = (void*)0;
            return 0;
        }
        if (gWl.compositor == (void*)0 || gWl.shm == (void*)0 || gWl.wmBase == (void*)0) {
            gWlLib.display_disconnect(d);
            gWl.display = (void*)0;
            return 0;
        }
        __wl_ensure_data_device();
        const char* env = getenv("SAFEC_GUI_SCALE");
        if (env != (const char*)0) {
            double v = atof(env);
            if (v >= 0.5 && v <= 8.0) { gWl.envScale = v; }
        }
        gWl.clip = String::new();
        __wl_init_app_id();
        int fds[2];
        if (pipe(&fds[0]) == 0) {
            fcntl(fds[0], 4 /* F_SETFL */, 2048 /* O_NONBLOCK */);
            fcntl(fds[1], 4, 2048);
            gWl.wakeR = fds[0];
            gWl.wakeW = fds[1];
        }
        gWlLib.display_roundtrip(d);   // the data device's current selection
    }
    gWlState = 1;
    return 1;
}

static int __wl_init() { return __wl_connect(); }

// ── the event pump ──────────────────────────────────────────────────────

static void __wl_lost() {
    // The compositor went away (or killed the connection over a protocol
    // error): tell every window, once.
    if (gWl.dead == 2) { return; }
    gWl.dead = 2;
    unsigned long i = 0UL;
    while (i < gui_backend_window_count()) {
        ?&GuiWindowState o = gui_backend_window_at(i);
        match (o) { case some(st): { gui_backend_post(st, GuiEvent::new(GUI_EVENT_CLOSE)); } case null: { } }
        i = i + 1UL;
    }
}

static int __wl_dispatch_pending() {
    unsafe {
        int r = gWlLib.display_dispatch_pending(gWl.display);
        if (r < 0) { __wl_lost(); return 0; }
        return r;
    }
}

// One read from the socket, waiting up to 'timeoutMs' (also returns when
// wake() writes to the pipe). Returns 1 if woken.
static int __wl_read(int timeoutMs) {
    unsafe {
        void* d = gWl.display;
        while (gWlLib.display_prepare_read(d) != 0) {
            if (__wl_dispatch_pending() == 0 && gWl.dead) { return 0; }
        }
        if (gWlLib.display_flush(d) < 0) {
            // EAGAIN only means the socket buffer is full; anything else is fatal
        }
        struct GuiWlPollFd fds[2];
        fds[0].fd = gWlLib.display_get_fd(d);
        fds[0].events = (short)1;
        fds[0].revents = (short)0;
        fds[1].fd = gWl.wakeR;
        fds[1].events = (short)1;
        fds[1].revents = (short)0;
        unsigned long n = gWl.wakeR >= 0 ? 2UL : 1UL;
        int r = poll((void*)&fds[0], n, timeoutMs);
        if (r > 0 && ((int)fds[0].revents & 1) != 0) {
            if (gWlLib.display_read_events(d) < 0) { __wl_lost(); }
        } else {
            gWlLib.display_cancel_read(d);
            if (r > 0 && ((int)fds[0].revents & (8 | 16)) != 0) { __wl_lost(); }   // POLLERR | POLLHUP
        }
        __wl_dispatch_pending();
        int woke = 0;
        if (gWl.wakeR >= 0 && ((int)fds[1].revents & 1) != 0) {
            char drain[64];
            while (read(gWl.wakeR, (void*)&drain[0], 64UL) > 0L) { woke = 1; }
        }
        return woke;
    }
}

static void __wl_flush_stale_frames() {
    unsafe {
        double now = __wl_now_ms();
        unsigned long i = 0UL;
        while (i < gui_backend_window_count()) {
            ?&GuiWindowState o = gui_backend_window_at(i);
            i = i + 1UL;
            if (o.is_null()) { continue; }
            struct GuiWindowState* st = (struct GuiWindowState*)o;
            struct GuiWlWindow* w = (struct GuiWlWindow*)st->native;
            if (w == (struct GuiWlWindow*)0 || w->pending == (struct GuiWlBuffer*)0) { continue; }
            if (w->frameCb != (void*)0 && now - w->frameAt > 250.0) {
                gWlLib.proxy_destroy(w->frameCb);
                w->frameCb = (void*)0;
                __wl_commit_pending(w);
            }
        }
    }
}

static int __wl_pump(int timeoutMs) {
    unsafe {
        if (gWlState != 1) { return 0; }
        if (gWl.dead) { return gui_backend_has_events(); }
        __wl_dispatch_pending();
        __wl_read(0);
        __wl_key_repeat();
        __wl_flush_stale_frames();
        if (gui_backend_has_events() || timeoutMs == 0) { return gui_backend_has_events(); }
        double start = __wl_now_ms();
        while (!gui_backend_has_events() && !gWl.dead) {
            double now = __wl_now_ms();
            int left = -1;
            if (timeoutMs > 0) {
                left = timeoutMs - (int)(now - start);
                if (left <= 0) { break; }
            }
            if (gWl.repeatKey != 0U) {
                int r = (int)(gWl.repeatNext - now + 0.999);
                if (r < 0) { r = 0; }
                if (left < 0 || r < left) { left = r; }
            }
            int woke = __wl_read(left);
            __wl_key_repeat();
            __wl_flush_stale_frames();
            if (woke) { break; }
        }
        return gui_backend_has_events();
    }
}

static void __wl_wake() {
    unsafe {
        if (gWl.wakeW >= 0) {
            char b = (char)1;
            write(gWl.wakeW, (const void*)&b, 1UL);
        }
    }
}

static long long __wl_now_ns() { return time_mono_ns(); }

static struct GuiNativeHandle __wl_native(&GuiWindowState st) {
    struct GuiNativeHandle h;
    h.kind = GUI_NATIVE_WAYLAND;
    unsafe { h.display = (unsigned long long)(unsigned long)gWl.display; }
    h.window = st.key;     // wl_surface*
    h.view = 0ULL;
    h.visual = 0ULL;
    return h;
}

?&GuiBackend gui_wayland_backend() {
    if (!gWlBackendInit) {
        gWlBackend.name = "wayland";
        gWlBackend.primaryMod = GUI_MOD_CTRL;
        gWlBackend.init = __wl_init;
        gWlBackend.window_create = __wl_create;
        gWlBackend.window_destroy = __wl_destroy;
        gWlBackend.window_present = __wl_present;
        gWlBackend.window_set_title = __wl_set_title;
        gWlBackend.window_set_size = __wl_set_size;
        gWlBackend.window_set_position = __wl_set_position;
        gWlBackend.window_update_limits = __wl_update_limits;
        gWlBackend.window_set_visible = __wl_set_visible;
        gWlBackend.window_focus = __wl_focus;
        gWlBackend.window_set_cursor = __wl_set_cursor;
        gWlBackend.pump = __wl_pump;
        gWlBackend.wake = __wl_wake;
        gWlBackend.now_ns = __wl_now_ns;
        gWlBackend.clipboard_set = __wl_clip_set;
        gWlBackend.clipboard_get = __wl_clip_get;
        gWlBackend.display_scale = __wl_display_scale;
        gWlBackend.native_handle = __wl_native;
        gWlBackendInit = 1;
    }
    return &gWlBackend;
}

int gui_wayland_available() { return __wl_connect(); }

int gui_use_wayland() {
    if (!gui_wayland_available()) { return 0; }
    ?&GuiBackend b = gui_wayland_backend();
    match (b) {
        case some(be): return gui_set_backend(be);
        case null: { }
    }
    return 0;
}

#else   // no Wayland on this target

?&GuiBackend gui_wayland_backend() { return null; }
int gui_wayland_available() { return 0; }
int gui_use_wayland() { return 0; }

#endif

} // namespace std
