// The Wayland GUI backend (std/gui/gui_wayland.sc) against a real
// compositor (run.sh starts sway and Weston headless).
//
//   wayland_check full     sway: frames read back with grim (full and
//                          partial presents, several in flight), input from
//                          a virtual pointer and keyboard this program
//                          creates on its own second connection (motion,
//                          buttons, wheel, keys, Shift, key repeat), the
//                          clipboard both ways with wl-copy / wl-paste,
//                          compositor and client resizes, hide/show, close,
//                          then two windows under one GuiApp.
//   wayland_check scaled   sway with output scale 1.5: fractional scaling
//                          (buffer size, frame read back, pointer coords).
//   wayland_check basic    any compositor: windows, presenting, resizing,
//                          two windows, closing.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/gui/gui_wayland.h>
#include <std/gui/gui_gpu.h>
#include <std/collections/string.h>

extern int system(const char* cmd);
extern void* fopen(const char* path, const char* mode);
extern unsigned long fread(void* buf, unsigned long size, unsigned long n, void* f);
extern int fclose(void* f);
extern int strcmp(const char* a, const char* b);
extern unsigned long strlen(const char* s);
extern char* getenv(const char* name);
extern double atof(const char* s);
extern int usleep(unsigned int us);
extern int unlink(const char* path);
extern int memfd_create(const char* name, unsigned int flags);
extern long write(int fd, const void* buf, unsigned long n);
extern int close(int fd);

// libwayland-client and libxkbcommon, linked (this is a test program).
extern void* wl_display_connect(const char* name);
extern void wl_display_disconnect(void* d);
extern int wl_display_roundtrip(void* d);
extern int wl_display_flush(void* d);
extern void* wl_proxy_marshal_flags(void* proxy, unsigned int opcode, const void* iface, unsigned int version, unsigned int flags, ...);
extern int wl_proxy_add_dispatcher(void* proxy, void* func, const void* data, void* udata);
extern unsigned int wl_proxy_get_version(void* proxy);
extern char wl_registry_interface;
extern char wl_seat_interface;
extern void* xkb_context_new(int flags);
extern void* xkb_keymap_new_from_names(void* ctx, const void* names, int flags);
extern char* xkb_keymap_get_as_string(void* keymap, int format);

// ── the injector: virtual pointer + keyboard on a second connection ─────

struct WlMessage { const char* name; const char* signature; const void** types; };
struct WlInterface { const char* name; int version; int nm; const WlMessage* m; int ne; const WlMessage* e; };

static const void* g_null_types[8];
static struct WlMessage g_vpm_msgs[3];
static struct WlMessage g_vp_msgs[9];
static struct WlMessage g_vkm_msgs[1];
static struct WlMessage g_vk_msgs[4];
static struct WlInterface g_vpm;    // zwlr_virtual_pointer_manager_v1
static struct WlInterface g_vp;     // zwlr_virtual_pointer_v1
static struct WlInterface g_vkm;    // zwp_virtual_keyboard_manager_v1
static struct WlInterface g_vk;     // zwp_virtual_keyboard_v1

static void msg(&WlMessage m, const char* name, const char* sig) {
    unsafe { m.name = name; m.signature = sig; m.types = &g_null_types[0]; }
}

static void* g_idpy = (void*)0;
static void* g_iseat = (void*)0;
static void* g_ivpm = (void*)0;
static void* g_ivkm = (void*)0;
static void* g_ptr = (void*)0;
static void* g_kbd = (void*)0;
static unsigned int g_time = 1000U;
static double g_extW = 640.0;
static double g_extH = 480.0;
static double g_scale = 1.0;      // the output's scale (run.sh)

static int reg_dispatch(const void* data, void* target, unsigned int opcode, const void* m, void* args) {
    unsafe {
        if (opcode != 0U) { return 0; }
        unsigned long long* a = (unsigned long long*)args;
        unsigned int name = *(unsigned int*)&a[0];
        const char* iface = *(const char**)&a[1];
        if (strcmp(iface, "wl_seat") == 0 && g_iseat == (void*)0) {
            g_iseat = wl_proxy_marshal_flags(target, 0U, (const void*)&wl_seat_interface, 1U, 0U, name, "wl_seat", 1U, (void*)0);
        } else if (strcmp(iface, "zwlr_virtual_pointer_manager_v1") == 0) {
            g_ivpm = wl_proxy_marshal_flags(target, 0U, (const void*)&g_vpm, 1U, 0U, name, g_vpm.name, 1U, (void*)0);
        } else if (strcmp(iface, "zwp_virtual_keyboard_manager_v1") == 0) {
            g_ivkm = wl_proxy_marshal_flags(target, 0U, (const void*)&g_vkm, 1U, 0U, name, g_vkm.name, 1U, (void*)0);
        }
    }
    return 0;
}

static int injector_open() {
    unsafe {
        msg(&g_vpm_msgs[0], "create_virtual_pointer", "?on");
        msg(&g_vpm_msgs[1], "destroy", "");
        msg(&g_vpm_msgs[2], "create_virtual_pointer_with_output", "2?o?on");
        g_vpm.name = "zwlr_virtual_pointer_manager_v1"; g_vpm.version = 1; g_vpm.nm = 3; g_vpm.m = &g_vpm_msgs[0];
        msg(&g_vp_msgs[0], "motion", "uff");
        msg(&g_vp_msgs[1], "motion_absolute", "uuuuu");
        msg(&g_vp_msgs[2], "button", "uuu");
        msg(&g_vp_msgs[3], "axis", "uuf");
        msg(&g_vp_msgs[4], "frame", "");
        msg(&g_vp_msgs[5], "axis_source", "u");
        msg(&g_vp_msgs[6], "axis_stop", "uu");
        msg(&g_vp_msgs[7], "axis_discrete", "uufi");
        msg(&g_vp_msgs[8], "destroy", "");
        g_vp.name = "zwlr_virtual_pointer_v1"; g_vp.version = 1; g_vp.nm = 9; g_vp.m = &g_vp_msgs[0];
        msg(&g_vkm_msgs[0], "create_virtual_keyboard", "on");
        g_vkm.name = "zwp_virtual_keyboard_manager_v1"; g_vkm.version = 1; g_vkm.nm = 1; g_vkm.m = &g_vkm_msgs[0];
        msg(&g_vk_msgs[0], "keymap", "uhu");
        msg(&g_vk_msgs[1], "key", "uuu");
        msg(&g_vk_msgs[2], "modifiers", "uuuu");
        msg(&g_vk_msgs[3], "destroy", "");
        g_vk.name = "zwp_virtual_keyboard_v1"; g_vk.version = 1; g_vk.nm = 4; g_vk.m = &g_vk_msgs[0];

        g_idpy = wl_display_connect((const char*)0);
        if (g_idpy == (void*)0) { return 0; }
        void* reg = wl_proxy_marshal_flags(g_idpy, 1U, (const void*)&wl_registry_interface, 1U, 0U, (void*)0);
        wl_proxy_add_dispatcher(reg, (void*)reg_dispatch, (const void*)0, (void*)0);
        wl_display_roundtrip(g_idpy);
        if (g_iseat == (void*)0 || g_ivpm == (void*)0 || g_ivkm == (void*)0) { return 0; }
        g_ptr = wl_proxy_marshal_flags(g_ivpm, 0U, (const void*)&g_vp, 1U, 0U, g_iseat, (void*)0);
        g_kbd = wl_proxy_marshal_flags(g_ivkm, 0U, (const void*)&g_vk, 1U, 0U, g_iseat, (void*)0);
        // a US keymap for the virtual keyboard
        void* ctx = xkb_context_new(0);
        void* km = xkb_keymap_new_from_names(ctx, (const void*)0, 0);
        char* text = xkb_keymap_get_as_string(km, 1);
        unsigned long n = strlen(text) + 1UL;
        int fd = memfd_create("keymap", 0U);
        write(fd, (const void*)text, n);
        wl_proxy_marshal_flags(g_kbd, 0U, (const void*)0, 1U, 0U, 1U, fd, (unsigned int)n);
        close(fd);
        wl_display_roundtrip(g_idpy);
        return 1;
    }
}

static void inj_flush() { unsafe { wl_display_roundtrip(g_idpy); } }

// Pointer to layout position (lx, ly) in logical units.
static void inj_move(double lx, double ly) {
    unsafe {
        g_time = g_time + 10U;
        wl_proxy_marshal_flags(g_ptr, 1U, (const void*)0, 1U, 0U, g_time, (unsigned int)(lx * 100.0),
                               (unsigned int)(ly * 100.0), (unsigned int)(g_extW * 100.0), (unsigned int)(g_extH * 100.0));
        wl_proxy_marshal_flags(g_ptr, 4U, (const void*)0, 1U, 0U);
        inj_flush();
    }
}

static void inj_button(unsigned int button, unsigned int state) {
    unsafe {
        g_time = g_time + 10U;
        wl_proxy_marshal_flags(g_ptr, 2U, (const void*)0, 1U, 0U, g_time, button, state);
        wl_proxy_marshal_flags(g_ptr, 4U, (const void*)0, 1U, 0U);
        inj_flush();
    }
}

static void inj_wheel(int steps) {
    unsafe {
        g_time = g_time + 10U;
        wl_proxy_marshal_flags(g_ptr, 5U, (const void*)0, 1U, 0U, 0U);   // axis_source wheel
        wl_proxy_marshal_flags(g_ptr, 7U, (const void*)0, 1U, 0U, g_time, 0U, steps * 15 * 256, steps);
        wl_proxy_marshal_flags(g_ptr, 4U, (const void*)0, 1U, 0U);
        inj_flush();
    }
}

static void inj_key(unsigned int key, unsigned int state) {
    unsafe {
        g_time = g_time + 10U;
        wl_proxy_marshal_flags(g_kbd, 1U, (const void*)0, 1U, 0U, g_time, key, state);
        inj_flush();
    }
}

static void inj_mods(unsigned int depressed) {
    unsafe {
        wl_proxy_marshal_flags(g_kbd, 2U, (const void*)0, 1U, 0U, depressed, 0U, 0U, 0U);
        inj_flush();
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

// Waits up to 'ms' for an event of 'kind' for 'win' (others are skipped).
static int wait_kind(&GuiWindow win, int kind, &GuiEvent out, int ms) {
    double start = gui_time_ms();
    while (gui_time_ms() - start < (double)ms) {
        if (win.wait_event(out, 20) && out.kind == kind) { return 1; }
    }
    return 0;
}

// MOUSE_ENTER or MOUSE_MOVE (the pointer's first position may come as either).
static int wait_pointer(&GuiWindow win, &GuiEvent out, int ms) {
    double start = gui_time_ms();
    while (gui_time_ms() - start < (double)ms) {
        if (win.wait_event(out, 20) && (out.kind == GUI_EVENT_MOUSE_MOVE || out.kind == GUI_EVENT_MOUSE_ENTER)) { return 1; }
    }
    return 0;
}

static void pump_ms(&GuiWindow win, int ms) {
    struct GuiEvent ev;
    double start = gui_time_ms();
    while (gui_time_ms() - start < (double)ms) { win.wait_event(&ev, 10); }
}

static int g_shotW = 0;
static int g_shotH = 0;
static unsigned char g_shot[640 * 480 * 3 * 4];

// The compositor's output, through grim (PPM).
static int grab() {
    unsafe {
        unlink("/tmp/safec-wl-shot.ppm");
        if (system("grim -t ppm /tmp/safec-wl-shot.ppm") != 0) { return 0; }
        // $SAFEC_WL_SHOTS=<dir>: keep every frame read back (debugging)
        if (getenv("SAFEC_WL_SHOTS") != (char*)0) { system("cp /tmp/safec-wl-shot.ppm \"$SAFEC_WL_SHOTS/shot-$(date +%s%N).ppm\""); }
        void* f = fopen("/tmp/safec-wl-shot.ppm", "rb");
        if (f == (void*)0) { return 0; }
        char head[64];
        // "P6\n<w> <h>\n255\n"
        unsigned long n = fread((void*)&head[0], 1UL, 15UL, f);
        int i = 3; int w = 0; int h = 0;
        while (head[i] >= '0' && head[i] <= '9') { w = w * 10 + (int)(head[i] - '0'); i = i + 1; }
        i = i + 1;
        while (head[i] >= '0' && head[i] <= '9') { h = h * 10 + (int)(head[i] - '0'); i = i + 1; }
        i = i + 5;   // "\n255\n"
        fclose(f);
        f = fopen("/tmp/safec-wl-shot.ppm", "rb");
        fread((void*)&head[0], 1UL, (unsigned long)i, f);
        unsigned long want = (unsigned long)w * (unsigned long)h * 3UL;
        if (want > 3686400UL) { fclose(f); return 0; }
        n = fread((void*)&g_shot[0], 1UL, want, f);
        fclose(f);
        g_shotW = w;
        g_shotH = h;
        return n == want ? 1 : 0;
    }
}

// 0xRRGGBB at output pixel (x, y) of the last grab.
static unsigned int shot_px(int x, int y) {
    if (x < 0 || y < 0 || x >= g_shotW || y >= g_shotH) { return 0xFFFFFFFFU; }
    unsigned long i = ((unsigned long)y * (unsigned long)g_shotW + (unsigned long)x) * 3UL;
    return ((unsigned int)g_shot[i] << 16) | ((unsigned int)g_shot[i + 1UL] << 8) | (unsigned int)g_shot[i + 2UL];
}

static int read_file(const char* path, &String out) {
    unsafe {
        out.clear();
        void* f = fopen(path, "rb");
        if (f == (void*)0) { return 0; }
        char buf[256];
        unsigned long n = fread((void*)&buf[0], 1UL, 255UL, f);
        fclose(f);
        out.push_n((const char*)&buf[0], n);
        return 1;
    }
}

static int file_exists(const char* path) {
    unsafe {
        void* f = fopen(path, "rb");
        if (f == (void*)0) { return 0; }
        fclose(f);
        return 1;
    }
}

static struct GuiColor RED() { return GuiColor::rgb((unsigned char)255, (unsigned char)0, (unsigned char)0); }
static struct GuiColor BLUE() { return GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)255); }
static struct GuiColor GREEN() { return GuiColor::rgb((unsigned char)0, (unsigned char)255, (unsigned char)0); }

// ── full (sway) ─────────────────────────────────────────────────────────

void t_backend() {
    &GuiBackend b = gui_backend();
    unsafe { ASSERT_EQ(strcmp(b.name, "wayland"), 0); }
    ASSERT_TRUE(gui_wayland_available());
}

void t_present_and_input() {
    unsafe {
        struct GuiWindowOptions o = GuiWindowOptions::new("wl first", 200, 150);
        struct GuiWindow win = GuiWindow::create_with(&o);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        ASSERT_TRUE(win.native_handle() != 0ULL);
        ASSERT_EQ(win.width, 200);
        ASSERT_EQ(win.height, 150);
        win.fill_rect(0, 0, 200, 150, RED());
        win.fill_rect(20, 20, 40, 30, BLUE());
        win.present();
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(5, 5), 0xFF0000ULL, "red background on screen", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(30, 30), 0x0000FFULL, "blue rectangle (R/B order)", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(195, 145), 0xFF0000ULL, "window is 200x150", __FILE__, __LINE__);

        // Partial presents back to back: each buffer must catch up on the
        // regions it missed.
        win.fill_rect(100, 100, 20, 20, GREEN());
        win.present_rect(100, 100, 20, 20);
        win.fill_rect(150, 20, 20, 20, GREEN());
        win.present_rect(150, 20, 20, 20);
        pump_ms(&win, 60);
        win.fill_rect(20, 100, 20, 20, GREEN());
        win.present_rect(20, 100, 20, 20);
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(110, 110), 0x00FF00ULL, "partial 1", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(160, 30), 0x00FF00ULL, "partial 2", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(30, 110), 0x00FF00ULL, "partial 3", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(30, 30), 0x0000FFULL, "unchanged region kept", __FILE__, __LINE__);
        int frame = 0;
        struct GuiEvent dummy;
        while (frame < 100) { win.present(); win.poll_event(&dummy); frame = frame + 1; }

        // ── input ──
        ASSERT_TRUE(injector_open());
        struct GuiEvent ev;
        inj_move(50.0, 40.0);
        ASSERT_TRUE(wait_pointer(&win, &ev, 1000));
        ASSERT_TRUE(ev.x > 49.0 && ev.x < 51.0 && ev.y > 39.0 && ev.y < 41.0);
        inj_move(60.0, 45.0);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_MOUSE_MOVE, &ev, 1000));
        ASSERT_TRUE(ev.x > 59.0 && ev.x < 61.0);
        inj_button(0x110U, 1U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_MOUSE_DOWN, &ev, 1000));
        ASSERT_EQ(ev.button, GUI_BUTTON_LEFT);
        ASSERT_TRUE(ev.x > 59.0 && ev.x < 61.0 && ev.y > 44.0 && ev.y < 46.0);
        inj_button(0x110U, 0U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_MOUSE_UP, &ev, 1000));
        inj_button(0x111U, 1U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_MOUSE_DOWN, &ev, 1000));
        ASSERT_EQ(ev.button, GUI_BUTTON_RIGHT);
        inj_button(0x111U, 0U);
        inj_wheel(1);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_SCROLL, &ev, 1000));
        ASSERT_TRUE(ev.scrollDy == -1.0);
        inj_wheel(-2);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_SCROLL, &ev, 1000));
        ASSERT_TRUE(ev.scrollDy == 2.0);
        win.set_cursor(GUI_CURSOR_HAND);
        win.set_cursor(GUI_CURSOR_IBEAM);
        win.set_cursor(GUI_CURSOR_HIDDEN);
        win.set_cursor(GUI_CURSOR_ARROW);

        pump_ms(&win, 50);
        ASSERT_TRUE(win.is_focused());
        inj_key(30U, 1U);                                    // a
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000));
        ASSERT_EQ(ev.keycode, 'A');
        ASSERT_EQ(ev.codepoint, 'a');
        ASSERT_EQ(ev.modifiers, 0);
        inj_key(30U, 0U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_UP, &ev, 1000));
        ASSERT_EQ(ev.keycode, 'A');
        inj_key(42U, 1U);                                    // Shift
        inj_mods(1U);
        inj_key(30U, 1U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000));   // Shift itself
        if (ev.keycode == GUI_KEY_SHIFT) { ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000)); }
        ASSERT_EQ(ev.keycode, 'A');
        ASSERT_EQ(ev.codepoint, 'A');
        ASSERT_EQ(ev.modifiers, GUI_MOD_SHIFT);
        inj_key(30U, 0U);
        inj_key(42U, 0U);
        inj_mods(0U);
        inj_key(29U, 1U);                                    // Ctrl+C: no text
        inj_mods(4U);
        inj_key(46U, 1U);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000));
        if (ev.keycode == GUI_KEY_CONTROL) { ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000)); }
        ASSERT_EQ(ev.keycode, 'C');
        ASSERT_EQ(ev.codepoint, 0);
        ASSERT_TRUE(ev.primary_mod());
        inj_key(46U, 0U);
        inj_key(29U, 0U);
        inj_mods(0U);
        inj_key(105U, 1U);                                   // Left
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000));
        ASSERT_EQ(ev.keycode, GUI_KEY_LEFT);
        ASSERT_EQ(ev.codepoint, 0);
        inj_key(105U, 0U);
        inj_key(28U, 1U);                                    // Enter
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_KEY_DOWN, &ev, 1000));
        ASSERT_EQ(ev.keycode, GUI_KEY_ENTER);
        inj_key(28U, 0U);
        // Key repeat (client-side, at the compositor's rate and delay).
        inj_key(48U, 1U);                                    // b, held
        int downs = 0;
        double start = gui_time_ms();
        while (gui_time_ms() - start < 1100.0) {
            if (win.wait_event(&ev, 20) && ev.kind == GUI_EVENT_KEY_DOWN && ev.codepoint == 'b') { downs = downs + 1; }
        }
        inj_key(48U, 0U);
        pump_ms(&win, 50);
        int after = 0;
        start = gui_time_ms();
        while (gui_time_ms() - start < 300.0) {
            if (win.wait_event(&ev, 20) && ev.kind == GUI_EVENT_KEY_DOWN) { after = after + 1; }
        }
        ASSERT_TRUE(downs >= 4);
        ASSERT_EQ(after, 0);

        // ── clipboard ──
        system("wl-copy 'from wl-copy \xC3\xA9' >/dev/null 2>&1");
        pump_ms(&win, 100);
        struct String s = String::new();
        ASSERT_TRUE(gui_clipboard_get_text(&s));
        ASSERT_TRUE(s.eq_cstr("from wl-copy \xC3\xA9"));
        // Taking the clipboard needs an input event newer than the current
        // selection (Wayland's rule; normally the Ctrl+C itself).
        inj_key(30U, 1U);
        inj_key(30U, 0U);
        pump_ms(&win, 50);
        ASSERT_TRUE(gui_clipboard_set_text("copied \xE2\x9C\x93"));
        ASSERT_TRUE(gui_clipboard_get_text(&s));
        ASSERT_TRUE(s.eq_cstr("copied \xE2\x9C\x93"));
        unlink("/tmp/safec-wl-paste.txt");
        unlink("/tmp/safec-wl-paste.done");
        system("sh -c 'wl-paste -n > /tmp/safec-wl-paste.txt; touch /tmp/safec-wl-paste.done' &");
        start = gui_time_ms();
        while (!file_exists("/tmp/safec-wl-paste.done") && gui_time_ms() - start < 3000.0) { pump_ms(&win, 20); }
        ASSERT_TRUE(read_file("/tmp/safec-wl-paste.txt", &s));
        ASSERT_TRUE(s.eq_cstr("copied \xE2\x9C\x93"));
        s.free();

        // ── sizes ──
        system("swaymsg -q '[title=\"wl first\"] resize set 260 170'");
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_RESIZE, &ev, 2000));
        ASSERT_EQ(ev.width, 260);
        ASSERT_EQ(win.width, 260);
        ASSERT_EQ(win.height, 170);
        win.fill_rect(0, 0, 260, 170, BLUE());
        win.present();
        system("swaymsg -q '[title=\"wl first\"] move position 0 0'");   // sway re-centers on resize
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(255, 165), 0x0000FFULL, "drawn at the compositor's size", __FILE__, __LINE__);
        win.set_size(220, 120);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_RESIZE, &ev, 1000));
        ASSERT_EQ(win.width, 220);
        ASSERT_EQ(win.height, 120);
        win.fill_rect(0, 0, 220, 120, GREEN());
        win.present();
        pump_ms(&win, 50);
        system("swaymsg -q '[title=\"wl first\"] move position 0 0'");
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(215, 115), 0x00FF00ULL, "drawn at our own size", __FILE__, __LINE__);
        ASSERT_TRUE(shot_px(240, 130) != 0x00FF00U);

        // ── hide / show ──
        win.hide();
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        ASSERT_TRUE(shot_px(100, 60) != 0x00FF00U);
        win.show();
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_EXPOSE, &ev, 2000));
        win.present();
        pump_ms(&win, 50);
        system("swaymsg -q '[title=\"wl first\"] move position 0 0'");
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(100, 60), 0x00FF00ULL, "shown again", __FILE__, __LINE__);

        // ── close ──
        system("swaymsg -q '[title=\"wl first\"] kill'");
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_CLOSE, &ev, 2000));
        ASSERT_EQ(win.shouldClose, 1);
        win.destroy();
        ASSERT_TRUE(win.platform.is_null());
    }
}

// ── several windows under one GuiApp ─────────────────────────────────────

static int g_clicks1 = 0;
static int g_clicks2 = 0;
static void click1(&Widget w, ?&void d) { g_clicks1 = g_clicks1 + 1; }
static void click2(&Widget w, ?&void d) { g_clicks2 = g_clicks2 + 1; }
static void paint2(&GuiApp app, &GuiWindow win, &GuiCanvas c, ?&void d) {
    c.set_fill_color(GuiColor::hex(0xFF0000U));
    c.fill_rect(0.0f, 0.0f, 8.0f, 8.0f);
}

static void click_at(&GuiApp app, double lx, double ly, &int counter, int want) {
    inj_move(lx, ly);
    int i = 0;
    while (i < 5) { app.pump(5); i = i + 1; }
    inj_button(0x110U, 1U);
    inj_button(0x110U, 0U);
    int tries = 0;
    while (*counter < want && tries < 200) { app.pump(5); tries = tries + 1; }
}

void t_multi() {
    unsafe {
        struct GuiApp app = GuiApp::new();
        ?&GuiWindow ow1 = app.open_window("wl first", 160, 100);
        ?&GuiWindow ow2 = app.open_window("wl second", 160, 100);
        ASSERT_FALSE(ow1.is_null());
        ASSERT_FALSE(ow2.is_null());
        if (ow1.is_null() || ow2.is_null()) { return; }
        &GuiWindow w1 = (struct GuiWindow*)ow1;
        &GuiWindow w2 = (struct GuiWindow*)ow2;
        ASSERT_TRUE(w1.native_handle() != w2.native_handle());
        &Widget r1 = Widget::vstack().set_padding(10);
        &Widget b1 = Widget::button("One").on_click(click1, (?&void)null);
        r1.add_child(b1);
        &Widget r2 = Widget::vstack().set_padding(10);
        &Widget b2 = Widget::button("Two").on_click(click2, (?&void)null);
        r2.add_child(b2);
        app.set_root(w1, r1);
        app.set_root(w2, r2);
        app.on_draw(w2, paint2, (?&void)null);
        int i = 0;
        while (i < 30) { app.pump(5); i = i + 1; }
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(322, 2), 0xFF0000ULL, "second window painted", __FILE__, __LINE__);
        ASSERT_TRUE(shot_px(2, 2) != 0xFF0000U);

        click_at(&app, 320.0 + (double)b2.x + 5.0, (double)b2.y + 5.0, &g_clicks2, 1);
        ASSERT_EQ(g_clicks2, 1);
        ASSERT_EQ(g_clicks1, 0);
        click_at(&app, (double)b1.x + 5.0, (double)b1.y + 5.0, &g_clicks1, 1);
        ASSERT_EQ(g_clicks1, 1);
        ASSERT_EQ(g_clicks2, 1);
        // keyboard focus followed the click: Space presses window 1's button
        i = 0;
        while (i < 10) { app.pump(5); i = i + 1; }
        inj_key(57U, 1U);
        inj_key(57U, 0U);
        i = 0;
        while (g_clicks1 < 2 && i < 200) { app.pump(5); i = i + 1; }
        ASSERT_EQ(g_clicks1, 2);
        ASSERT_EQ(g_clicks2, 1);

        w2.set_title("Second \xE2\x9C\x93");
        i = 0;
        while (i < 5) { app.pump(5); i = i + 1; }
        ASSERT_EQ(system("swaymsg -t get_tree | grep -q 'Second \xE2\x9C\x93'"), 0);

        system("swaymsg -q '[title=\"Second \xE2\x9C\x93\"] kill'");
        i = 0;
        while (app.window_count() > 1UL && i < 400) { app.pump(5); i = i + 1; }
        ASSERT_EQ(app.window_count(), 1UL);
        ASSERT_EQ(app.pump(0), 1);
        app.free();
        ASSERT_EQ(app.window_count(), 0UL);
    }
}

// ── scaled (sway, output scale 1.5) ─────────────────────────────────────

void t_scaled() {
    unsafe {
        struct GuiWindowOptions o = GuiWindowOptions::new("wl first", 200, 150);
        struct GuiWindow win = GuiWindow::create_with(&o);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        struct GuiEvent ev;
        // The preferred scale arrives once the surface is on the output.
        win.present();
        int got = 0;
        double start = gui_time_ms();
        while (win.scale != g_scale && gui_time_ms() - start < 2000.0) { win.wait_event(&ev, 20); }
        ASSERT_TRUE(win.scale == g_scale);
        ASSERT_EQ(win.width, (int)(200.0 * g_scale + 0.5));
        ASSERT_EQ(win.height, (int)(150.0 * g_scale + 0.5));
        ASSERT_EQ(win.logical_width(), 200);
        win.fill_rect(0, 0, win.width, win.height, RED());
        win.fill_rect(30, 30, 30, 30, BLUE());
        win.present();
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        // the output is 640x480 device pixels; the window at layout (0,0)
        test_assert_eq_u(shot_px(5, 5), 0xFF0000ULL, "scaled: background", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(45, 45), 0x0000FFULL, "scaled: buffer pixels 1:1", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(295, 220), 0xFF0000ULL, "scaled: 300x225 device pixels", __FILE__, __LINE__);
        ASSERT_TRUE(injector_open());
        inj_move(40.0, 20.0);
        ASSERT_TRUE(wait_pointer(&win, &ev, 1000));
        ASSERT_TRUE(ev.x > 59.0 && ev.x < 61.0 && ev.y > 29.0 && ev.y < 31.0);
        win.destroy();

        // hiDpi off: scale 1, the compositor scales the window up
        struct GuiWindowOptions o2 = GuiWindowOptions::new("wl first", 100, 80);
        o2.hiDpi = 0;
        struct GuiWindow lo = GuiWindow::create_with(&o2);
        lo.present();
        pump_ms(&lo, 200);
        ASSERT_TRUE(lo.scale == 1.0);
        ASSERT_EQ(lo.width, 100);
        lo.destroy();
    }
}

// ── gl (sway): the EGL presenter on Wayland ─────────────────────────────

void t_gl() {
    unsafe {
        struct GuiWindowOptions o = GuiWindowOptions::new("wl first", 200, 150);
        o.present = GUI_PRESENT_GPU;
        struct GuiWindow win = GuiWindow::create_with(&o);
        ASSERT_FALSE(win.platform.is_null());
        if (win.platform.is_null()) return;
        ASSERT_TRUE(gui_window_uses_gpu(&win));
        ASSERT_EQ(strcmp(gui_window_presenter_name(&win), "gl"), 0);
        struct GuiEvent ev;
        win.present();
        double start = gui_time_ms();
        while (win.scale != g_scale && gui_time_ms() - start < 2000.0) { win.wait_event(&ev, 20); }
        ASSERT_TRUE(win.scale == g_scale);
        int W = win.width; int H = win.height;
        int s10 = (int)(10.0 * g_scale);
        win.fill_rect(0, 0, W, H, RED());
        win.fill_rect(2 * s10, 2 * s10, 4 * s10, 3 * s10, BLUE());
        win.present();
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(5, 5), 0xFF0000ULL, "gl: background", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(3 * s10, 3 * s10), 0x0000FFULL, "gl: rectangle, not flipped", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(W - 3, H - 3), 0xFF0000ULL, "gl: buffer size 1:1", __FILE__, __LINE__);
        ASSERT_TRUE(shot_px(W + 3, H + 3) != 0xFF0000U);
        win.fill_rect(10 * s10, 10 * s10, 2 * s10, 2 * s10, GREEN());
        win.present_rect(10 * s10, 10 * s10, 2 * s10, 2 * s10);
        pump_ms(&win, 150);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(11 * s10, 11 * s10), 0x00FF00ULL, "gl: partial update", __FILE__, __LINE__);
        test_assert_eq_u(shot_px(3 * s10, 3 * s10), 0x0000FFULL, "gl: rest kept", __FILE__, __LINE__);
        int frame = 0;
        while (frame < 60) { win.present(); win.poll_event(&ev); frame = frame + 1; }
        win.set_size(240, 120);
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_RESIZE, &ev, 1000));
        ASSERT_EQ(win.width, (int)(240.0 * g_scale + 0.5));
        win.fill_rect(0, 0, win.width, win.height, GREEN());
        win.present();
        pump_ms(&win, 100);
        system("swaymsg -q '[title=\"wl first\"] move position 0 0'");
        pump_ms(&win, 100);
        ASSERT_TRUE(grab());
        test_assert_eq_u(shot_px(win.width - 3, win.height - 3), 0x00FF00ULL, "gl: resized", __FILE__, __LINE__);
        // hidden and shown again
        win.hide();
        pump_ms(&win, 100);
        win.present();      // ignored while hidden
        win.show();
        ASSERT_TRUE(wait_kind(&win, GUI_EVENT_EXPOSE, &ev, 2000));
        win.present();
        pump_ms(&win, 100);
        ASSERT_TRUE(gui_window_uses_gpu(&win));
        win.destroy();
    }
}

// ── basic (any compositor) ──────────────────────────────────────────────

void t_basic() {
    unsafe {
        struct GuiApp app = GuiApp::new();
        ?&GuiWindow ow1 = app.open_window("wl first", 160, 100);
        ?&GuiWindow ow2 = app.open_window("wl second", 120, 90);
        ASSERT_FALSE(ow1.is_null());
        ASSERT_FALSE(ow2.is_null());
        if (ow1.is_null() || ow2.is_null()) { return; }
        &GuiWindow w1 = (struct GuiWindow*)ow1;
        &GuiWindow w2 = (struct GuiWindow*)ow2;
        app.on_draw(w2, paint2, (?&void)null);
        app.set_root(w1, Widget::button("Hello"));
        int i = 0;
        while (i < 20) { app.pump(5); i = i + 1; }
        ASSERT_TRUE(w1.width >= 160 && w2.width >= 120);
        w1.set_title("renamed");
        w2.set_size(150, 110);
        i = 0;
        while (w2.width != (int)(150.0 * w2.scale + 0.5) && i < 100) { app.pump(5); i = i + 1; }
        ASSERT_EQ(w2.logical_width(), 150);
        w1.set_min_size(100, 80);
        w1.set_cursor(GUI_CURSOR_IBEAM);
        struct String s = String::new();
        gui_clipboard_get_text(&s);      // no selection: must not hang or crash
        s.free();
        w2.request_close();
        i = 0;
        while (app.window_count() > 1UL && i < 100) { app.pump(5); i = i + 1; }
        ASSERT_EQ(app.window_count(), 1UL);
        app.free();
        ASSERT_EQ(app.window_count(), 0UL);
    }
}

int main(int argc, char** argv) {
    struct TestSuite t = TestSuite::new();
    unsafe {
        const char* mode = "full";
        if (argc > 1) { mode = argv[1]; }
        const char* sc = getenv("SAFEC_WL_OUTPUT_SCALE");
        if (sc != (const char*)0) { g_scale = atof(sc); }
        g_extW = 640.0 / g_scale;
        g_extH = 480.0 / g_scale;
        t.add("wayland: selected for $WAYLAND_DISPLAY", t_backend);
        if (strcmp(mode, "full") == 0) {
            t.add("wayland: present, partial presents, input, clipboard, sizes, hide, close", t_present_and_input);
            t.add("wayland: two windows under GuiApp", t_multi);
        } else if (strcmp(mode, "scaled") == 0) {
            t.add("wayland: fractional scale", t_scaled);
            t.add("wayland: OpenGL ES presenter (EGL), scaled", t_gl);
        } else if (strcmp(mode, "gl") == 0) {
            t.add("wayland: OpenGL ES presenter (EGL)", t_gl);
        } else {
            t.add("wayland: windows, sizes, two windows, close", t_basic);
        }
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
