// SafeC Standard Library — GUI: GPU presentation, the portable part (see
// gui_gpu.h). The presenters themselves are per platform:
// gui_gpu_gl_x11.sc (EGL/OpenGL ES, X11 + Wayland), gui_gpu_metal_cocoa.sc,
// gui_gpu_d3d11_win32.sc, gui_gpu_web_wasm.sc (WebGL); gui_gpu_none_fb.sc
// has none.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
extern char* getenv(const char* name);
#elif defined(__wasm__)
// std/gui/gui_web.js: the page's 'present' option (1 webgl, 2 2d, 0 unset).
extern int std_gui_web_present_mode();
#endif

namespace std {

static int __gui_gpu_env_is(const char* v, const char* want) {
    unsafe {
        if (v == (const char*)0) { return 0; }
        unsigned long i = 0UL;
        while (want[i] != '\0' && v[i] == want[i]) { i = i + 1UL; }
        return (want[i] == '\0' && v[i] == '\0') ? 1 : 0;
    }
}

int gui_gpu_wanted(int present) {
    if (present == GUI_PRESENT_GPU) { return 1; }
    if (present == GUI_PRESENT_CPU) { return 0; }
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    unsafe { return __gui_gpu_env_is(getenv("SAFEC_GUI_PRESENT"), "gpu"); }
#elif defined(__wasm__)
    unsafe { return std_gui_web_present_mode() == 1 ? 1 : 0; }
#else
    return 0;
#endif
}

int gui_window_use_gpu(&GuiWindow win) {
    if (gui_window_state(win).is_null()) { return 0; }
    if (gui_window_uses_gpu(win)) { return 1; }
    ?&GuiPresenter p = gui_gpu_presenter();
    if (p.is_null()) { return 0; }
    return gui_window_set_presenter(win, p);
}

void gui_window_use_cpu(&GuiWindow win) {
    gui_window_set_presenter(win, null);
}

int gui_window_uses_gpu(const &GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return 0;
        case some(st): { return st.presenter.is_null() ? 0 : 1; }
    }
    return 0;
}

const char* gui_window_presenter_name(const &GuiWindow win) {
    ?&GuiWindowState ost = gui_window_state(win);
    match (ost) {
        case null: return "cpu";
        case some(st): {
            unsafe {
                if (st.presenter.is_null()) { return "cpu"; }
                struct GuiPresenter* p = (struct GuiPresenter*)st.presenter;
                return p->name;
            }
        }
    }
    return "cpu";
}

} // namespace std
