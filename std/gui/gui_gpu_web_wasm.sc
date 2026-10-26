// SafeC Standard Library — GUI: the WebGL presenter for the WebAssembly
// <canvas> backend (see gui_gpu.h and gui_web.h). The work is in
// std/gui/gui_web.js: the window's canvas gets a WebGL2 context (else
// WebGL1), the premultiplied frame is uploaded into a texture straight
// from linear memory — the damaged rectangle with UNPACK_ROW_LENGTH on
// WebGL2, the damaged rows on WebGL1 — and drawn as one quad; no straight-
// alpha conversion is needed, unlike putImageData. A lost context, or no
// WebGL at all, sends the window back to putImageData on a fresh canvas.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>

namespace std {

// ── imports (std/gui/gui_web.js) ────────────────────────────────────────
extern int  std_gui_web_gl_attach(int id);
extern int  std_gui_web_gl_present(int id, const unsigned char* pixels, int bufW, int bufH, int x, int y, int w, int h);
extern void std_gui_web_gl_detach(int id);

static struct GuiPresenter gWebGlPresenter;
static int gWebGlPresenterInit = 0;

static int __webgl_attach(&GuiWindowState st, struct GuiNativeHandle h) {
    if (h.kind != GUI_NATIVE_WEB) { return 0; }
    unsafe { return std_gui_web_gl_attach((int)st.id); }
}

static void __webgl_detach(&GuiWindowState st) {
    unsafe { std_gui_web_gl_detach((int)st.id); }
}

static void __webgl_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                            int x, int y, int rw, int rh) {
    unsafe {
        if (std_gui_web_gl_present((int)st.id, (const unsigned char*)px, bw, bh, x, y, rw, rh)) { return; }
    }
    // the context is gone: putImageData from now on
    __webgl_detach(st);
    st.presenter = null;
    st.presenterData = null;
    &GuiBackend b = gui_backend();
    b.window_present(st, px, bw, bh, 0, 0, bw, bh);
}

?&GuiPresenter gui_gpu_presenter() {
    if (!gWebGlPresenterInit) {
        gWebGlPresenter.name = "webgl";
        gWebGlPresenter.attach = __webgl_attach;
        gWebGlPresenter.present = __webgl_present;
        gWebGlPresenter.detach = __webgl_detach;
        gWebGlPresenterInit = 1;
    }
    return &gWebGlPresenter;
}

} // namespace std
