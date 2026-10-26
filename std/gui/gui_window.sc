// SafeC Standard Library — GUI: the portable window layer (see gui.h and
// gui_backend.h). GuiWindow's methods, the event queues, the clipboard and
// the clock, all on top of whichever struct GuiBackend is in use — this
// file contains no platform code.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/mem.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
extern void* memset(void* ptr, int val, size_t n);
extern void* memcpy(void* dst, const void* src, size_t n);

namespace std {

// ── backend selection ─────────────────────────────────────────────────────

static ?&GuiBackend gGuiInstalled = null;
static int gGuiInitState = 0;          // 0 = not tried, 1 = ready, -1 = init failed
static unsigned int gGuiNextId = 1U;
static struct Vec<&GuiWindowState> gGuiWindows;
static int gGuiWindowsInit = 0;

static &GuiBackend __gui_be_raw() {
    match (gGuiInstalled) {
        case some(b): return b;
        case null: { }
    }
    return gui_platform_backend();
}

&GuiBackend gui_backend() {
    gui_backend_ready();
    return __gui_be_raw();
}

int gui_backend_ready() {
    &GuiBackend b = __gui_be_raw();
    if (gGuiInitState == 0) {
        gGuiInitState = b.init() ? 1 : -1;
    }
    return gGuiInitState == 1 ? 1 : 0;
}

int gui_set_backend(&GuiBackend b) {
    if (gui_backend_window_count() > 0UL) { return 0; }
    gGuiInstalled = b;
    gGuiInitState = 0;
    return 1;
}

static void __gui_windows_init() {
    if (!gGuiWindowsInit) {
        gGuiWindows = Vec::new();
        gGuiWindowsInit = 1;
    }
}

unsigned long gui_backend_window_count() {
    if (!gGuiWindowsInit) { return 0UL; }
    return gGuiWindows.len;
}

?&GuiWindowState gui_backend_window_at(unsigned long i) {
    if (!gGuiWindowsInit || i >= gGuiWindows.len) { return null; }
    return *gGuiWindows.at(i);
}

?&GuiWindowState gui_backend_find(unsigned long long key) {
    if (!gGuiWindowsInit || key == 0ULL) { return null; }
    unsigned long i = 0UL;
    while (i < gGuiWindows.len) {
        &GuiWindowState st = *gGuiWindows.at(i);
        if (st.key == key) { return st; }
        i = i + 1UL;
    }
    return null;
}

static unsigned long __gui_queue_pending(&GuiWindowState st) {
    return st.queue.len - st.qhead;
}

int gui_backend_has_events() {
    if (!gGuiWindowsInit) { return 0; }
    unsigned long i = 0UL;
    while (i < gGuiWindows.len) {
        &GuiWindowState st = *gGuiWindows.at(i);
        if (__gui_queue_pending(st) > 0UL) { return 1; }
        i = i + 1UL;
    }
    return 0;
}

double gui_time_ms() {
    &GuiBackend b = __gui_be_raw();
    return (double)b.now_ns() / 1000000.0;
}

void gui_backend_post(&GuiWindowState st, struct GuiEvent ev) {
    ev.windowId = st.id;
    ev.defaultPrevented = 0;
    if (ev.timeMs == 0.0) { ev.timeMs = gui_time_ms(); }
    if (ev.kind == GUI_EVENT_MOUSE_DOWN) {
        // Two presses of the same button within 500 ms and 5 pixels of each
        // other make a double click, a third one a triple click.
        double dx = ev.x - st.lastDownX;
        double dy = ev.y - st.lastDownY;
        if (dx < 0.0) { dx = -dx; }
        if (dy < 0.0) { dy = -dy; }
        if (st.clickCount > 0 && ev.button == st.lastDownButton &&
            ev.timeMs - st.lastDownMs <= 500.0 && dx <= 5.0 * st.scale && dy <= 5.0 * st.scale) {
            st.clickCount = st.clickCount + 1;
            if (st.clickCount > 3) { st.clickCount = 1; }
        } else {
            st.clickCount = 1;
        }
        st.lastDownMs = ev.timeMs;
        st.lastDownX = ev.x;
        st.lastDownY = ev.y;
        st.lastDownButton = ev.button;
        ev.clicks = st.clickCount;
    }
    if (ev.kind == GUI_EVENT_MOUSE_ENTER) { st.hovered = 1; }
    if (ev.kind == GUI_EVENT_MOUSE_LEAVE) { st.hovered = 0; }
    st.queue.push(ev);
}

static int __gui_queue_pop(&GuiWindowState st, &GuiEvent out) {
    if (st.qhead >= st.queue.len) {
        st.queue.clear();
        st.qhead = 0UL;
        return 0;
    }
    *out = *st.queue.at(st.qhead);
    st.qhead = st.qhead + 1UL;
    if (st.qhead >= st.queue.len) {
        st.queue.clear();
        st.qhead = 0UL;
    }
    return 1;
}

?&GuiWindowState gui_window_state(const &GuiWindow win) {
    unsafe {
        if (win.platform.is_null()) { return null; }
        return (struct GuiWindowState*)win.platform;
    }
}

// ── events, options ───────────────────────────────────────────────────────

struct GuiEvent GuiEvent::new(int kind) {
    struct GuiEvent e;
    unsafe { memset((void*)&e, 0, sizeof(struct GuiEvent)); }
    e.kind = kind;
    return e;
}

void GuiEvent::prevent_default() {
    self.defaultPrevented = 1;
}

int GuiEvent::has_mod(int mod) const {
    return (self.modifiers & mod) != 0 ? 1 : 0;
}

int GuiEvent::primary_mod() const {
    return (self.modifiers & gui_primary_mod()) != 0 ? 1 : 0;
}

int gui_primary_mod() {
    &GuiBackend b = __gui_be_raw();
    return b.primaryMod;
}

double gui_display_scale() {
    if (!gui_backend_ready()) { return 1.0; }
    &GuiBackend b = gui_backend();
    double s = b.display_scale();
    return s > 0.0 ? s : 1.0;
}

struct GuiWindowOptions GuiWindowOptions::new(const &char title, int width, int height) {
    struct GuiWindowOptions o;
    o.title = title;
    o.x = GUI_POS_DEFAULT;
    o.y = GUI_POS_DEFAULT;
    o.width = width;
    o.height = height;
    o.minWidth = 0;
    o.minHeight = 0;
    o.maxWidth = 0;
    o.maxHeight = 0;
    o.resizable = 1;
    o.decorated = 1;
    o.visible = 1;
    o.hiDpi = 1;
    o.topmost = 0;
    return o;
}

// ── creation / destruction ────────────────────────────────────────────────

static &heap unsigned char __gui_alloc_pixels(int w, int h) {
    unsigned long n = (unsigned long)(w > 0 ? w : 1) * (unsigned long)(h > 0 ? h : 1) * 4UL;
    unsafe { return (&heap unsigned char)alloc_zeroed_raw_(n); }
}

static struct GuiWindow __gui_window_blank() {
    struct GuiWindow win;
    win.platform = null;
    unsafe { win.pixels = (&heap unsigned char)0; }
    win.width = 0;
    win.height = 0;
    win.shouldClose = 0;
    win.scale = 1.0;
    win.id = 0U;
    return win;
}

struct GuiWindow GuiWindow::offscreen(int width, int height) {
    struct GuiWindow win = __gui_window_blank();
    if (width < 1) { width = 1; }
    if (height < 1) { height = 1; }
    win.pixels = __gui_alloc_pixels(width, height);
    win.width = width;
    win.height = height;
    return win;
}

struct GuiWindow GuiWindow::create(const &char title, int width, int height) {
    struct GuiWindowOptions o = GuiWindowOptions::new(title, width, height);
    o.hiDpi = 0;
    return GuiWindow::create_with(&o);
}

struct GuiWindow GuiWindow::create_with(const &GuiWindowOptions opts) {
    struct GuiWindow win = __gui_window_blank();
    if (!gui_backend_ready()) { return win; }
    &GuiBackend b = gui_backend();
    __gui_windows_init();

    &GuiWindowState st;
    unsafe {
        st = (struct GuiWindowState*)alloc_zeroed_raw_(sizeof(struct GuiWindowState));
    }
    st.id = gGuiNextId;
    st.key = 0ULL;
    st.native = null;
    st.presenterData = null;
    st.presenter = null;
    st.queue = Vec::new();
    st.qhead = 0UL;
    unsafe {
        if (opts.title != (const char*)0) { st.title = String::from(opts.title); }
        else { st.title = String::new(); }
    }
    st.x = opts.x;
    st.y = opts.y;
    st.logicalW = opts.width > 0 ? opts.width : 1;
    st.logicalH = opts.height > 0 ? opts.height : 1;
    st.scale = 1.0;
    st.pixelW = st.logicalW;
    st.pixelH = st.logicalH;
    st.hiDpi = opts.hiDpi;
    st.minW = opts.minWidth; st.minH = opts.minHeight;
    st.maxW = opts.maxWidth; st.maxH = opts.maxHeight;
    st.visible = opts.visible;
    st.focused = 0;
    st.hovered = 0;
    st.cursor = GUI_CURSOR_ARROW;
    st.resizable = opts.resizable;
    st.decorated = opts.decorated;
    st.topmost = opts.topmost;
    st.clickCount = 0;
    st.lastDownMs = -1000000.0;

    if (!b.window_create(st, opts)) {
        st.queue.free();
        st.title.free();
        unsafe { dealloc_raw_((void*)st); }
        return win;
    }
    gGuiNextId = gGuiNextId + 1U;
    if (st.scale <= 0.0) { st.scale = 1.0; }
    if (st.pixelW < 1) { st.pixelW = 1; }
    if (st.pixelH < 1) { st.pixelH = 1; }
    gGuiWindows.push(st);

    win.id = st.id;
    win.scale = st.scale;
    win.width = st.pixelW;
    win.height = st.pixelH;
    win.pixels = __gui_alloc_pixels(win.width, win.height);
    unsafe { win.platform = (void*)st; }
    return win;
}

static void __gui_detach_presenter(&GuiWindowState st) {
    unsafe {
        if (!st.presenter.is_null()) {
            struct GuiPresenter* p = (struct GuiPresenter*)st.presenter;
            if (p->detach != (fn void(&GuiWindowState))0) { p->detach(st); }
            st.presenter = null;
            st.presenterData = null;
        }
    }
}

void GuiWindow::destroy() { self.drop(); }

void GuiWindow::drop() {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case some(st): {
            __gui_detach_presenter(st);
            &GuiBackend b = gui_backend();
            b.window_destroy(st);
            unsigned long i = 0UL;
            while (i < gGuiWindows.len) {
                &GuiWindowState other = *gGuiWindows.at(i);
                unsafe {
                    if ((void*)other == (void*)st) { gGuiWindows.remove(i); break; }
                }
                i = i + 1UL;
            }
            st.queue.free();
            st.title.free();
            unsafe { dealloc_raw_((void*)st); }
        }
        case null: { }
    }
    unsafe {
        if ((void*)self.pixels != (void*)0) { dealloc_raw_((void*)self.pixels); }
        self.pixels = (&heap unsigned char)0;
    }
    self.platform = null;
    self.width = 0;
    self.height = 0;
}

// ── events ────────────────────────────────────────────────────────────────

// Brings the pixel buffer to the size the backend reported.
static void __gui_window_apply_size(&GuiWindow win, &GuiWindowState st, int w, int h) {
    win.scale = st.scale;
    if (w < 1 || h < 1) { return; }
    if (w == win.width && h == win.height) { return; }
    &heap unsigned char px = __gui_alloc_pixels(w, h);
    unsafe {
        if ((void*)win.pixels != (void*)0) { dealloc_raw_((void*)win.pixels); }
    }
    win.pixels = px;
    win.width = w;
    win.height = h;
}

static int __gui_window_take(&GuiWindow win, &GuiWindowState st, &GuiEvent out) {
    while (__gui_queue_pop(st, out)) {
        int k = out.kind;
        if (k == GUI_EVENT_CLOSE) { win.shouldClose = 1; }
        if (k == GUI_EVENT_RESIZE || k == GUI_EVENT_SCALE) {
            if (out.width < 1 || out.height < 1) { continue; }   // minimized
            int same = (out.width == win.width && out.height == win.height) ? 1 : 0;
            __gui_window_apply_size(win, st, out.width, out.height);
            if (same && k == GUI_EVENT_RESIZE) { continue; }     // nothing changed
        }
        return 1;
    }
    return 0;
}

int GuiWindow::poll_event(&GuiEvent outEvent) {
    outEvent.kind = GUI_EVENT_NONE;
    outEvent.defaultPrevented = 0;
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return 0;
        case some(st): {
            if (__gui_queue_pending(st) == 0UL) {
                &GuiBackend b = gui_backend();
                b.pump(0);
            }
            return __gui_window_take(self, st, outEvent);
        }
    }
    return 0;
}

int GuiWindow::wait_event(&GuiEvent outEvent, int timeoutMs) {
    outEvent.kind = GUI_EVENT_NONE;
    outEvent.defaultPrevented = 0;
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return 0;
        case some(st): {
            &GuiBackend b = gui_backend();
            if (__gui_queue_pending(st) == 0UL) { b.pump(0); }
            if (__gui_queue_pending(st) == 0UL) {
                double start = gui_time_ms();
                while (__gui_queue_pending(st) == 0UL) {
                    int left = timeoutMs;
                    if (timeoutMs >= 0) {
                        left = timeoutMs - (int)(gui_time_ms() - start);
                        if (left <= 0) { break; }
                    }
                    b.pump(left);
                }
            }
            return __gui_window_take(self, st, outEvent);
        }
    }
    return 0;
}

// ── presentation ──────────────────────────────────────────────────────────

void GuiWindow::present() {
    self.present_rect(0, 0, self.width, self.height);
}

void GuiWindow::present_rect(int x, int y, int w, int h) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            if (x < 0) { w = w + x; x = 0; }
            if (y < 0) { h = h + y; y = 0; }
            if (x + w > self.width) { w = self.width - x; }
            if (y + h > self.height) { h = self.height - y; }
            if (w <= 0 || h <= 0) { return; }
            unsafe {
                const unsigned char* px = (const unsigned char*)self.pixels;
                if (!st.presenter.is_null()) {
                    struct GuiPresenter* p = (struct GuiPresenter*)st.presenter;
                    p->present(st, px, self.width, self.height, x, y, w, h);
                    return;
                }
            }
            &GuiBackend b = gui_backend();
            unsafe {
                b.window_present(st, (const unsigned char*)self.pixels, self.width, self.height, x, y, w, h);
            }
        }
    }
}

int gui_window_set_presenter(&GuiWindow win, ?&GuiPresenter p) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return 0;
        case some(st): {
            __gui_detach_presenter(st);
            match (p) {
                case null: return 1;
                case some(pp): {
                    &GuiBackend b = gui_backend();
                    struct GuiNativeHandle h = b.native_handle(st);
                    unsafe { st.presenter = (void*)pp; }
                    if (!pp.attach(st, h)) {
                        st.presenter = null;
                        st.presenterData = null;
                        return 0;
                    }
                    return 1;
                }
            }
        }
    }
    return 0;
}

// ── pixels ────────────────────────────────────────────────────────────────

void GuiWindow::set_pixel(int x, int y, unsigned int rgba) {
    if (x < 0 || y < 0 || x >= self.width || y >= self.height) { return; }
    unsafe {
        unsigned long i = ((unsigned long)y * (unsigned long)self.width + (unsigned long)x) * 4UL;
        self.pixels[i + 0UL] = (unsigned char)((rgba >> 24) & 0xFFU);
        self.pixels[i + 1UL] = (unsigned char)((rgba >> 16) & 0xFFU);
        self.pixels[i + 2UL] = (unsigned char)((rgba >> 8) & 0xFFU);
        self.pixels[i + 3UL] = (unsigned char)(rgba & 0xFFU);
    }
}

unsigned int GuiWindow::get_pixel(int x, int y) const {
    if (x < 0 || y < 0 || x >= self.width || y >= self.height) { return 0U; }
    unsafe {
        unsigned long i = ((unsigned long)y * (unsigned long)self.width + (unsigned long)x) * 4UL;
        return ((unsigned int)self.pixels[i] << 24) | ((unsigned int)self.pixels[i + 1UL] << 16) |
               ((unsigned int)self.pixels[i + 2UL] << 8) | (unsigned int)self.pixels[i + 3UL];
    }
}

void GuiWindow::clear(struct GuiColor c) {
    unsigned int a = (unsigned int)c.a;
    unsigned int r = ((unsigned int)c.r * a + 127U) / 255U;
    unsigned int g = ((unsigned int)c.g * a + 127U) / 255U;
    unsigned int b = ((unsigned int)c.b * a + 127U) / 255U;
    unsafe {
        unsigned int* p = (unsigned int*)self.pixels;
        unsigned long n = (unsigned long)self.width * (unsigned long)self.height;
        // Memory order R,G,B,A: as a little-endian word that's A<<24|B<<16|G<<8|R.
        unsigned char bytes[4];
        bytes[0] = (unsigned char)r; bytes[1] = (unsigned char)g; bytes[2] = (unsigned char)b; bytes[3] = (unsigned char)a;
        unsigned int word = *(unsigned int*)&bytes[0];
        unsigned long i = 0UL;
        while (i < n) { p[i] = word; i = i + 1UL; }
    }
}

// ── window management ────────────────────────────────────────────────────

void GuiWindow::set_title(const &char title) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            st.title.free();
            st.title = String::from(title);
            &GuiBackend b = gui_backend();
            b.window_set_title(st, title);
        }
    }
}

void GuiWindow::set_size(int width, int height) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            if (width < 1) { width = 1; }
            if (height < 1) { height = 1; }
            &GuiBackend b = gui_backend();
            b.window_set_size(st, width, height);
        }
    }
}

void GuiWindow::set_position(int x, int y) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            &GuiBackend b = gui_backend();
            b.window_set_position(st, x, y);
            // what was asked for until the platform reports where it went
            st.x = x;
            st.y = y;
        }
    }
}

int GuiWindow::pos_x() const {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) { case null: return GUI_POS_DEFAULT; case some(st): return st.x; }
    return GUI_POS_DEFAULT;
}

int GuiWindow::pos_y() const {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) { case null: return GUI_POS_DEFAULT; case some(st): return st.y; }
    return GUI_POS_DEFAULT;
}

void GuiWindow::set_min_size(int width, int height) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            st.minW = width; st.minH = height;
            &GuiBackend b = gui_backend();
            b.window_update_limits(st);
        }
    }
}

void GuiWindow::set_max_size(int width, int height) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            st.maxW = width; st.maxH = height;
            &GuiBackend b = gui_backend();
            b.window_update_limits(st);
        }
    }
}

static void __gui_window_visible(&GuiWindow win, int v) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return;
        case some(st): {
            st.visible = v;
            &GuiBackend b = gui_backend();
            b.window_set_visible(st, v);
        }
    }
}

void GuiWindow::show() { __gui_window_visible(self, 1); }
void GuiWindow::hide() { __gui_window_visible(self, 0); }

int GuiWindow::is_visible() const {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) { case null: return 0; case some(st): return st.visible; }
    return 0;
}

void GuiWindow::focus() {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            &GuiBackend b = gui_backend();
            b.window_focus(st);
        }
    }
}

int GuiWindow::is_focused() const {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) { case null: return 0; case some(st): return st.focused; }
    return 0;
}

void GuiWindow::set_cursor(int cursor) {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): {
            if (cursor < 0 || cursor >= GUI_CURSOR_COUNT) { cursor = GUI_CURSOR_ARROW; }
            if (st.cursor == cursor) { return; }
            st.cursor = cursor;
            &GuiBackend b = gui_backend();
            b.window_set_cursor(st, cursor);
        }
    }
}

int GuiWindow::logical_width() const {
    double s = self.scale > 0.0 ? self.scale : 1.0;
    return (int)((double)self.width / s + 0.5);
}

int GuiWindow::logical_height() const {
    double s = self.scale > 0.0 ? self.scale : 1.0;
    return (int)((double)self.height / s + 0.5);
}

void GuiWindow::request_close() {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return;
        case some(st): { gui_backend_post(st, GuiEvent::new(GUI_EVENT_CLOSE)); }
    }
}

unsigned long long GuiWindow::native_handle() const {
    ?&GuiWindowState ost = gui_window_state(self);
    match (ost) {
        case null: return 0ULL;
        case some(st): {
            &GuiBackend b = gui_backend();
            struct GuiNativeHandle h = b.native_handle(st);
            return h.window;
        }
    }
    return 0ULL;
}

// ── clipboard ─────────────────────────────────────────────────────────────

int gui_clipboard_set_text(const &char utf8) {
    if (!gui_backend_ready()) { return 0; }
    &GuiBackend b = gui_backend();
    unsigned long n = 0UL;
    unsafe { while (utf8[n] != '\0') { n = n + 1UL; } }
    return b.clipboard_set(utf8, n);
}

int gui_clipboard_get_text(&String out) {
    out.clear();
    if (!gui_backend_ready()) { return 0; }
    &GuiBackend b = gui_backend();
    return b.clipboard_get(out);
}

} // namespace std
