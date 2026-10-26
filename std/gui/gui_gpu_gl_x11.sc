// SafeC Standard Library — GUI: the OpenGL ES presenter, through EGL, for
// the X11 and Wayland backends (see gui_gpu.h).
//
// libEGL.so.1, libGLESv2.so.2 and (on Wayland) libwayland-egl.so.1 are
// dlopen()ed on first use. Per window: an EGL window surface (the X11
// Window itself; a wl_egl_window over the wl_surface on Wayland), an
// OpenGL ES 2 context, one RGBA texture the size of the buffer and a
// program drawing it as a full-window quad. present() uploads only the
// damaged rows (the damaged rectangle itself when the context is ES 3,
// which has GL_UNPACK_ROW_LENGTH), redraws the quad and swaps — with
// EGL_KHR/EXT_swap_buffers_with_damage when offered, so the compositor or X
// server only updates that rectangle. Premultiplied RGBA is what GL and
// both window systems expect, so no conversion is needed.
// safeguard: link linux -ldl
// safeguard: deb-recommends linux libegl1 libgles2
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>
#include <std/mem.h>
extern void* memset(void* ptr, int val, size_t n);
extern void* dlopen(const char* file, int mode);
extern void* dlsym(void* handle, const char* name);
extern char* getenv(const char* name);

namespace std {

// ── EGL / GLES (dlopen) ─────────────────────────────────────────────────

typedef fn void*(const char*) EglGetProcAddress_;
typedef fn void*(void*) EglGetDisplay_;
typedef fn void*(unsigned int, void*, const void*) EglGetPlatformDisplay_;
typedef fn unsigned int(void*, int*, int*) EglInitialize_;
typedef fn unsigned int(unsigned int) EglBindAPI_;
typedef fn unsigned int(void*, const int*, void**, int, int*) EglChooseConfig_;
typedef fn unsigned int(void*, void*, int, int*) EglGetConfigAttrib_;
typedef fn void*(void*, void*, void*, const int*) EglCreateContext_;
typedef fn void*(void*, void*, void*, const int*) EglCreateWindowSurface_;
typedef fn unsigned int(void*, void*, void*, void*) EglMakeCurrent_;
typedef fn unsigned int(void*, void*) EglSwapBuffers_;
typedef fn unsigned int(void*, void*, const int*, int) EglSwapWithDamage_;
typedef fn unsigned int(void*, int) EglSwapInterval_;
typedef fn unsigned int(void*, void*) EglDestroy_;
typedef fn unsigned int(void*, void*, int, int*) EglQuerySurface_;
typedef fn const char*(void*, int) EglQueryString_;
typedef fn int() EglGetError_;

typedef fn unsigned int(unsigned int) GlCreateShader_;
typedef fn void(unsigned int, int, const char**, const int*) GlShaderSource_;
typedef fn void(unsigned int) GlUInt1_;
typedef fn void(unsigned int, unsigned int, int*) GlGetiv_;
typedef fn unsigned int() GlCreateProgram_;
typedef fn void(unsigned int, unsigned int) GlUInt2_;
typedef fn void(unsigned int, unsigned int, const char*) GlBindAttribLocation_;
typedef fn int(unsigned int, const char*) GlGetUniformLocation_;
typedef fn void(int, int) GlUniform1i_;
typedef fn void(int, unsigned int*) GlGenTextures_;
typedef fn void(int, const unsigned int*) GlDeleteTextures_;
typedef fn void(unsigned int, unsigned int, int) GlTexParameteri_;
typedef fn void(unsigned int, int, int, int, int, int, unsigned int, unsigned int, const void*) GlTexImage2D_;
typedef fn void(unsigned int, int, int, int, int, int, unsigned int, unsigned int, const void*) GlTexSubImage2D_;
typedef fn void(unsigned int, int) GlPixelStorei_;
typedef fn void(int, int, int, int) GlViewport_;
typedef fn void(float, float, float, float) GlClearColor_;
typedef fn void(unsigned int, int, unsigned int, unsigned char, int, const void*) GlVertexAttribPointer_;
typedef fn void(unsigned int, int, int) GlDrawArrays_;
typedef fn unsigned int() GlGetError_;
typedef fn const unsigned char*(unsigned int) GlGetString_;

typedef fn void*(void*, int, int) WlEglCreate_;
typedef fn void(void*, int, int, int, int) WlEglResize_;
typedef fn void(void*) WlEglDestroy_;

struct GuiGlLib {
    int loaded;    // 0 not tried, 1 ok, -1 unavailable
    EglGetProcAddress_ getProcAddress;
    EglGetDisplay_ getDisplay;
    EglGetPlatformDisplay_ getPlatformDisplay;      // EGL 1.5, or the EXT; may be null
    EglInitialize_ initialize;
    EglBindAPI_ bindAPI;
    EglChooseConfig_ chooseConfig;
    EglGetConfigAttrib_ getConfigAttrib;
    EglCreateContext_ createContext;
    EglCreateWindowSurface_ createWindowSurface;
    EglMakeCurrent_ makeCurrent;
    EglSwapBuffers_ swapBuffers;
    EglSwapInterval_ swapInterval;
    EglDestroy_ destroySurface;
    EglDestroy_ destroyContext;
    EglQuerySurface_ querySurface;
    EglQueryString_ queryString;
    EglGetError_ getError;

    GlCreateShader_ createShader;
    GlShaderSource_ shaderSource;
    GlUInt1_ compileShader;
    GlGetiv_ getShaderiv;
    GlCreateProgram_ createProgram;
    GlUInt2_ attachShader;
    GlBindAttribLocation_ bindAttribLocation;
    GlUInt1_ linkProgram;
    GlGetiv_ getProgramiv;
    GlUInt1_ useProgram;
    GlUInt1_ deleteShader;
    GlUInt1_ deleteProgram;
    GlGetUniformLocation_ getUniformLocation;
    GlUniform1i_ uniform1i;
    GlGenTextures_ genTextures;
    GlDeleteTextures_ deleteTextures;
    GlUInt2_ bindTexture;
    GlUInt1_ activeTexture;
    GlTexParameteri_ texParameteri;
    GlTexImage2D_ texImage2D;
    GlTexSubImage2D_ texSubImage2D;
    GlPixelStorei_ pixelStorei;
    GlViewport_ viewport;
    GlClearColor_ clearColor;
    GlUInt1_ clear;
    GlUInt1_ disable;
    GlVertexAttribPointer_ vertexAttribPointer;
    GlUInt1_ enableVertexAttribArray;
    GlDrawArrays_ drawArrays;
    GlGetError_ glGetError;
    GlGetString_ getString;

    int wlEgl;
    WlEglCreate_ wlEglCreate;
    WlEglResize_ wlEglResize;
    WlEglDestroy_ wlEglDestroy;
};

static struct GuiGlLib gGl;

#define GUI_EGL_NONE 0x3038
#define GUI_EGL_SUCCESS 0x3000
#define GUI_EGL_OPENGL_ES_API 0x30A0
#define GUI_EGL_EXTENSIONS 0x3055
#define GUI_EGL_WIDTH 0x3057
#define GUI_EGL_HEIGHT 0x3056
#define GUI_EGL_NATIVE_VISUAL_ID 0x302E
#define GUI_EGL_PLATFORM_X11 0x31D5
#define GUI_EGL_PLATFORM_WAYLAND 0x31D8

#define GUI_GL_TEXTURE_2D 0x0DE1
#define GUI_GL_RGBA 0x1908
#define GUI_GL_UNSIGNED_BYTE 0x1401
#define GUI_GL_FLOAT 0x1406
#define GUI_GL_TRIANGLE_STRIP 0x0005
#define GUI_GL_COLOR_BUFFER_BIT 0x00004000U
#define GUI_GL_BLEND 0x0BE2
#define GUI_GL_DEPTH_TEST 0x0B71
#define GUI_GL_TEXTURE0 0x84C0
#define GUI_GL_UNPACK_ALIGNMENT 0x0CF5
#define GUI_GL_UNPACK_ROW_LENGTH 0x0CF2
#define GUI_GL_UNPACK_SKIP_ROWS 0x0CF3
#define GUI_GL_UNPACK_SKIP_PIXELS 0x0CF4
#define GUI_GL_VERSION 0x1F02

static void* __gl_sym(void* lib, const char* name, &int ok) {
    unsafe {
        void* p = (void*)0;
        if (lib != (void*)0) { p = dlsym(lib, name); }
        if (p == (void*)0 && gGl.getProcAddress != (EglGetProcAddress_)0) { p = gGl.getProcAddress(name); }
        if (p == (void*)0) { *ok = 0; }
        return p;
    }
}

static int __gl_load() {
    if (gGl.loaded != 0) { return gGl.loaded > 0 ? 1 : 0; }
    gGl.loaded = -1;
    unsafe {
        void* egl = dlopen("libEGL.so.1", 2);
        if (egl == (void*)0) { egl = dlopen("libEGL.so", 2); }
        if (egl == (void*)0) { return 0; }
        void* gles = dlopen("libGLESv2.so.2", 2);
        if (gles == (void*)0) { gles = dlopen("libGLESv2.so", 2); }
        int ok = 1;
        gGl.getProcAddress = (EglGetProcAddress_)__gl_sym(egl, "eglGetProcAddress", &ok);
        gGl.getDisplay = (EglGetDisplay_)__gl_sym(egl, "eglGetDisplay", &ok);
        gGl.initialize = (EglInitialize_)__gl_sym(egl, "eglInitialize", &ok);
        gGl.bindAPI = (EglBindAPI_)__gl_sym(egl, "eglBindAPI", &ok);
        gGl.chooseConfig = (EglChooseConfig_)__gl_sym(egl, "eglChooseConfig", &ok);
        gGl.getConfigAttrib = (EglGetConfigAttrib_)__gl_sym(egl, "eglGetConfigAttrib", &ok);
        gGl.createContext = (EglCreateContext_)__gl_sym(egl, "eglCreateContext", &ok);
        gGl.createWindowSurface = (EglCreateWindowSurface_)__gl_sym(egl, "eglCreateWindowSurface", &ok);
        gGl.makeCurrent = (EglMakeCurrent_)__gl_sym(egl, "eglMakeCurrent", &ok);
        gGl.swapBuffers = (EglSwapBuffers_)__gl_sym(egl, "eglSwapBuffers", &ok);
        gGl.swapInterval = (EglSwapInterval_)__gl_sym(egl, "eglSwapInterval", &ok);
        gGl.destroySurface = (EglDestroy_)__gl_sym(egl, "eglDestroySurface", &ok);
        gGl.destroyContext = (EglDestroy_)__gl_sym(egl, "eglDestroyContext", &ok);
        gGl.querySurface = (EglQuerySurface_)__gl_sym(egl, "eglQuerySurface", &ok);
        gGl.queryString = (EglQueryString_)__gl_sym(egl, "eglQueryString", &ok);
        gGl.getError = (EglGetError_)__gl_sym(egl, "eglGetError", &ok);
        if (!ok) { return 0; }
        gGl.getPlatformDisplay = (EglGetPlatformDisplay_)dlsym(egl, "eglGetPlatformDisplay");
        if (gGl.getPlatformDisplay == (EglGetPlatformDisplay_)0) {
            gGl.getPlatformDisplay = (EglGetPlatformDisplay_)gGl.getProcAddress("eglGetPlatformDisplayEXT");
        }
        gGl.createShader = (GlCreateShader_)__gl_sym(gles, "glCreateShader", &ok);
        gGl.shaderSource = (GlShaderSource_)__gl_sym(gles, "glShaderSource", &ok);
        gGl.compileShader = (GlUInt1_)__gl_sym(gles, "glCompileShader", &ok);
        gGl.getShaderiv = (GlGetiv_)__gl_sym(gles, "glGetShaderiv", &ok);
        gGl.createProgram = (GlCreateProgram_)__gl_sym(gles, "glCreateProgram", &ok);
        gGl.attachShader = (GlUInt2_)__gl_sym(gles, "glAttachShader", &ok);
        gGl.bindAttribLocation = (GlBindAttribLocation_)__gl_sym(gles, "glBindAttribLocation", &ok);
        gGl.linkProgram = (GlUInt1_)__gl_sym(gles, "glLinkProgram", &ok);
        gGl.getProgramiv = (GlGetiv_)__gl_sym(gles, "glGetProgramiv", &ok);
        gGl.useProgram = (GlUInt1_)__gl_sym(gles, "glUseProgram", &ok);
        gGl.deleteShader = (GlUInt1_)__gl_sym(gles, "glDeleteShader", &ok);
        gGl.deleteProgram = (GlUInt1_)__gl_sym(gles, "glDeleteProgram", &ok);
        gGl.getUniformLocation = (GlGetUniformLocation_)__gl_sym(gles, "glGetUniformLocation", &ok);
        gGl.uniform1i = (GlUniform1i_)__gl_sym(gles, "glUniform1i", &ok);
        gGl.genTextures = (GlGenTextures_)__gl_sym(gles, "glGenTextures", &ok);
        gGl.deleteTextures = (GlDeleteTextures_)__gl_sym(gles, "glDeleteTextures", &ok);
        gGl.bindTexture = (GlUInt2_)__gl_sym(gles, "glBindTexture", &ok);
        gGl.activeTexture = (GlUInt1_)__gl_sym(gles, "glActiveTexture", &ok);
        gGl.texParameteri = (GlTexParameteri_)__gl_sym(gles, "glTexParameteri", &ok);
        gGl.texImage2D = (GlTexImage2D_)__gl_sym(gles, "glTexImage2D", &ok);
        gGl.texSubImage2D = (GlTexSubImage2D_)__gl_sym(gles, "glTexSubImage2D", &ok);
        gGl.pixelStorei = (GlPixelStorei_)__gl_sym(gles, "glPixelStorei", &ok);
        gGl.viewport = (GlViewport_)__gl_sym(gles, "glViewport", &ok);
        gGl.clearColor = (GlClearColor_)__gl_sym(gles, "glClearColor", &ok);
        gGl.clear = (GlUInt1_)__gl_sym(gles, "glClear", &ok);
        gGl.disable = (GlUInt1_)__gl_sym(gles, "glDisable", &ok);
        gGl.vertexAttribPointer = (GlVertexAttribPointer_)__gl_sym(gles, "glVertexAttribPointer", &ok);
        gGl.enableVertexAttribArray = (GlUInt1_)__gl_sym(gles, "glEnableVertexAttribArray", &ok);
        gGl.drawArrays = (GlDrawArrays_)__gl_sym(gles, "glDrawArrays", &ok);
        gGl.glGetError = (GlGetError_)__gl_sym(gles, "glGetError", &ok);
        gGl.getString = (GlGetString_)__gl_sym(gles, "glGetString", &ok);
        if (!ok) { return 0; }
        void* wle = dlopen("libwayland-egl.so.1", 2);
        if (wle != (void*)0) {
            int wo = 1;
            gGl.wlEglCreate = (WlEglCreate_)__gl_sym(wle, "wl_egl_window_create", &wo);
            gGl.wlEglResize = (WlEglResize_)__gl_sym(wle, "wl_egl_window_resize", &wo);
            gGl.wlEglDestroy = (WlEglDestroy_)__gl_sym(wle, "wl_egl_window_destroy", &wo);
            gGl.wlEgl = wo;
        }
    }
    gGl.loaded = 1;
    return 1;
}

// ── displays (one EGLDisplay per native display) ────────────────────────

#define GUI_GL_MAX_DISPLAYS 4

struct GuiGlDisplay {
    unsigned long long native;
    int kind;
    void* dpy;
    int swapDamage;              // 0 none, 1 KHR, 2 EXT
    EglSwapWithDamage_ swapWithDamage;
};

static struct GuiGlDisplay gGlDisplays[GUI_GL_MAX_DISPLAYS];
static int gGlDisplayCount = 0;

static int __gl_has_ext(const char* list, const char* name) {
    unsafe {
        if (list == (const char*)0) { return 0; }
        unsigned long i = 0UL;
        while (list[i] != '\0') {
            unsigned long k = 0UL;
            while (name[k] != '\0' && list[i + k] == name[k]) { k = k + 1UL; }
            if (name[k] == '\0' && (list[i + k] == ' ' || list[i + k] == '\0') && (i == 0UL || list[i - 1UL] == ' ')) { return 1; }
            i = i + 1UL;
        }
        return 0;
    }
}

static struct GuiGlDisplay* __gl_display(struct GuiNativeHandle h) {
    unsafe {
        int i = 0;
        while (i < gGlDisplayCount) {
            if (gGlDisplays[i].native == h.display && gGlDisplays[i].kind == h.kind) { return &gGlDisplays[i]; }
            i = i + 1;
        }
        if (gGlDisplayCount >= GUI_GL_MAX_DISPLAYS) { return (struct GuiGlDisplay*)0; }
        void* native = (void*)(unsigned long)h.display;
        void* dpy = (void*)0;
        if (gGl.getPlatformDisplay != (EglGetPlatformDisplay_)0) {
            unsigned int platform = h.kind == GUI_NATIVE_WAYLAND ? GUI_EGL_PLATFORM_WAYLAND : GUI_EGL_PLATFORM_X11;
            dpy = gGl.getPlatformDisplay(platform, native, (const void*)0);
        }
        if (dpy == (void*)0) { dpy = gGl.getDisplay(native); }
        if (dpy == (void*)0) { return (struct GuiGlDisplay*)0; }
        int major = 0; int minor = 0;
        if (!gGl.initialize(dpy, &major, &minor)) { return (struct GuiGlDisplay*)0; }
        struct GuiGlDisplay* d = &gGlDisplays[gGlDisplayCount];
        memset((void*)d, 0, sizeof(struct GuiGlDisplay));
        d->native = h.display;
        d->kind = h.kind;
        d->dpy = dpy;
        const char* ext = gGl.queryString(dpy, GUI_EGL_EXTENSIONS);
        if (__gl_has_ext(ext, "EGL_KHR_swap_buffers_with_damage")) {
            d->swapWithDamage = (EglSwapWithDamage_)gGl.getProcAddress("eglSwapBuffersWithDamageKHR");
        } else if (__gl_has_ext(ext, "EGL_EXT_swap_buffers_with_damage")) {
            d->swapWithDamage = (EglSwapWithDamage_)gGl.getProcAddress("eglSwapBuffersWithDamageEXT");
        }
        gGlDisplayCount = gGlDisplayCount + 1;
        return d;
    }
}

// ── per window ──────────────────────────────────────────────────────────

struct GuiGlWindow {
    struct GuiGlDisplay* d;
    void* config;
    void* ctx;
    void* surface;
    void* wlEgl;          // wl_egl_window (Wayland)
    int kind;
    unsigned int program;
    unsigned int texture;
    int texW; int texH;
    int es3;              // GL_UNPACK_ROW_LENGTH available
    int eglW; int eglH;   // wl_egl_window size
};

static struct GuiPresenter gGlPresenter;
static int gGlPresenterInit = 0;

static unsigned int __gl_shader(unsigned int type, const char* src) {
    unsafe {
        unsigned int s = gGl.createShader(type);
        const char* srcs[1];
        srcs[0] = src;
        gGl.shaderSource(s, 1, &srcs[0], (const int*)0);
        gGl.compileShader(s);
        int ok = 0;
        gGl.getShaderiv(s, 0x8B81U /* GL_COMPILE_STATUS */, &ok);
        if (!ok) { gGl.deleteShader(s); return 0U; }
        return s;
    }
}

static void __gl_free(struct GuiGlWindow* w) {
    unsafe {
        void* dpy = w->d != (struct GuiGlDisplay*)0 ? w->d->dpy : (void*)0;
        if (dpy != (void*)0 && w->ctx != (void*)0) {
            gGl.makeCurrent(dpy, w->surface, w->surface, w->ctx);
            if (w->texture != 0U) { gGl.deleteTextures(1, &w->texture); }
            if (w->program != 0U) { gGl.deleteProgram(w->program); }
            gGl.makeCurrent(dpy, (void*)0, (void*)0, (void*)0);
        }
        if (dpy != (void*)0 && w->surface != (void*)0) { gGl.destroySurface(dpy, w->surface); }
        if (dpy != (void*)0 && w->ctx != (void*)0) { gGl.destroyContext(dpy, w->ctx); }
        if (w->wlEgl != (void*)0) { gGl.wlEglDestroy(w->wlEgl); }
        dealloc_raw_((void*)w);
    }
}

// The config: RGB(A)8, window-capable, ES2-renderable; on X11 the one
// whose visual is the window's.
static void* __gl_config(void* dpy, struct GuiNativeHandle h) {
    unsafe {
        int attribs[16];
        int n = 0;
        attribs[n] = 0x3033; attribs[n + 1] = 0x0004; n = n + 2;   // SURFACE_TYPE: WINDOW_BIT
        attribs[n] = 0x3040; attribs[n + 1] = 0x0004; n = n + 2;   // RENDERABLE_TYPE: ES2_BIT
        attribs[n] = 0x3024; attribs[n + 1] = 8; n = n + 2;        // RED
        attribs[n] = 0x3023; attribs[n + 1] = 8; n = n + 2;        // GREEN
        attribs[n] = 0x3022; attribs[n + 1] = 8; n = n + 2;        // BLUE
        if (h.kind == GUI_NATIVE_WAYLAND) { attribs[n] = 0x3021; attribs[n + 1] = 8; n = n + 2; }   // ALPHA
        attribs[n] = GUI_EGL_NONE;
        void* configs[64];
        int count = 0;
        if (!gGl.chooseConfig(dpy, &attribs[0], &configs[0], 64, &count) || count <= 0) { return (void*)0; }
        if (h.kind == GUI_NATIVE_X11 && h.visual != 0ULL) {
            // Visual's second field is its id (Xlib's struct Visual).
            unsigned long vid = *(unsigned long*)((unsigned long)h.visual + 8UL);
            int i = 0;
            while (i < count) {
                int v = 0;
                gGl.getConfigAttrib(dpy, configs[i], GUI_EGL_NATIVE_VISUAL_ID, &v);
                if ((unsigned long)v == vid) { return configs[i]; }
                i = i + 1;
            }
        }
        return configs[0];
    }
}

static int __gl_attach(&GuiWindowState st, struct GuiNativeHandle h) {
    unsafe {
        if (h.kind != GUI_NATIVE_X11 && h.kind != GUI_NATIVE_WAYLAND) { return 0; }
        if (h.display == 0ULL || h.window == 0ULL) { return 0; }
        if (!__gl_load()) { return 0; }
        if (h.kind == GUI_NATIVE_WAYLAND && !gGl.wlEgl) { return 0; }
        struct GuiGlDisplay* d = __gl_display(h);
        if (d == (struct GuiGlDisplay*)0) { return 0; }
        if (!gGl.bindAPI(GUI_EGL_OPENGL_ES_API)) { return 0; }
        struct GuiGlWindow* w = (struct GuiGlWindow*)alloc_zeroed_raw_(sizeof(struct GuiGlWindow));
        w->d = d;
        w->kind = h.kind;
        w->config = __gl_config(d->dpy, h);
        if (w->config == (void*)0) { __gl_free(w); return 0; }
        int ctxAttribs[3];
        ctxAttribs[0] = 0x3098; ctxAttribs[1] = 2; ctxAttribs[2] = GUI_EGL_NONE;   // CONTEXT_CLIENT_VERSION 2
        w->ctx = gGl.createContext(d->dpy, w->config, (void*)0, &ctxAttribs[0]);
        if (w->ctx == (void*)0) { __gl_free(w); return 0; }
        void* native = (void*)(unsigned long)h.window;      // the X11 Window (an XID)
        if (h.kind == GUI_NATIVE_WAYLAND) {
            w->eglW = st.pixelW > 0 ? st.pixelW : 1;
            w->eglH = st.pixelH > 0 ? st.pixelH : 1;
            w->wlEgl = gGl.wlEglCreate((void*)(unsigned long)h.window, w->eglW, w->eglH);
            if (w->wlEgl == (void*)0) { __gl_free(w); return 0; }
            native = w->wlEgl;
        }
        w->surface = gGl.createWindowSurface(d->dpy, w->config, native, (const int*)0);
        if (w->surface == (void*)0) { __gl_free(w); return 0; }
        if (!gGl.makeCurrent(d->dpy, w->surface, w->surface, w->ctx)) { __gl_free(w); return 0; }
        // vsync on X11; a Wayland compositor paces frames itself, and an
        // interval of 1 there would block a hidden window's swap forever.
        gGl.swapInterval(d->dpy, h.kind == GUI_NATIVE_WAYLAND ? 0 : 1);

        const char* vs = "attribute vec2 p;\nvarying vec2 uv;\nvoid main() {\n  uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);\n  gl_Position = vec4(p, 0.0, 1.0);\n}\n";
        const char* fs = "precision mediump float;\nvarying vec2 uv;\nuniform sampler2D t;\nvoid main() { gl_FragColor = texture2D(t, uv); }\n";
        unsigned int v = __gl_shader(0x8B31U /* VERTEX */, vs);
        unsigned int f = __gl_shader(0x8B30U /* FRAGMENT */, fs);
        if (v == 0U || f == 0U) { __gl_free(w); return 0; }
        unsigned int prog = gGl.createProgram();
        gGl.attachShader(prog, v);
        gGl.attachShader(prog, f);
        gGl.bindAttribLocation(prog, 0U, "p");
        gGl.linkProgram(prog);
        gGl.deleteShader(v);
        gGl.deleteShader(f);
        int linked = 0;
        gGl.getProgramiv(prog, 0x8B82U /* LINK_STATUS */, &linked);
        if (!linked) { gGl.deleteProgram(prog); __gl_free(w); return 0; }
        w->program = prog;
        gGl.useProgram(prog);
        gGl.uniform1i(gGl.getUniformLocation(prog, "t"), 0);
        gGl.genTextures(1, &w->texture);
        gGl.activeTexture(GUI_GL_TEXTURE0);
        gGl.bindTexture(GUI_GL_TEXTURE_2D, w->texture);
        gGl.texParameteri(GUI_GL_TEXTURE_2D, 0x2801 /* MIN_FILTER */, 0x2600 /* NEAREST */);
        gGl.texParameteri(GUI_GL_TEXTURE_2D, 0x2800 /* MAG_FILTER */, 0x2600);
        gGl.texParameteri(GUI_GL_TEXTURE_2D, 0x2802 /* WRAP_S */, 0x812F /* CLAMP_TO_EDGE */);
        gGl.texParameteri(GUI_GL_TEXTURE_2D, 0x2803 /* WRAP_T */, 0x812F);
        gGl.pixelStorei(GUI_GL_UNPACK_ALIGNMENT, 4);
        gGl.disable(GUI_GL_BLEND);
        gGl.disable(GUI_GL_DEPTH_TEST);
        const unsigned char* ver = gGl.getString(GUI_GL_VERSION);
        // "OpenGL ES 3.x ..."
        if (ver != (const unsigned char*)0 && ver[0] == (unsigned char)'O' && ver[10] >= (unsigned char)'3' && ver[10] <= (unsigned char)'9') { w->es3 = 1; }
        if (gGl.glGetError() != 0U) { __gl_free(w); return 0; }
        st.presenterData = (void*)w;
        return 1;
    }
}

static void __gl_detach(&GuiWindowState st) {
    unsafe {
        if (st.presenterData.is_null()) { return; }
        __gl_free((struct GuiGlWindow*)st.presenterData);
        st.presenterData = null;
    }
}

// Something failed for good: show this frame the software way and leave
// the window to it from now on.
static void __gl_fallback(&GuiWindowState st, const &unsigned char px, int bw, int bh) {
    __gl_detach(st);
    st.presenter = null;
    &GuiBackend b = gui_backend();
    b.window_present(st, px, bw, bh, 0, 0, bw, bh);
}

static void __gl_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                         int x, int y, int rw, int rh) {
    unsafe {
        struct GuiGlWindow* w = (struct GuiGlWindow*)st.presenterData;
        if (w == (struct GuiGlWindow*)0) { return; }
        if (!st.visible) { return; }   // (Wayland: no buffer may be attached to an unmapped surface)
        void* dpy = w->d->dpy;
        if (w->wlEgl != (void*)0 && (w->eglW != bw || w->eglH != bh)) {
            gGl.wlEglResize(w->wlEgl, bw, bh, 0, 0);
            w->eglW = bw;
            w->eglH = bh;
        }
        if (!gGl.makeCurrent(dpy, w->surface, w->surface, w->ctx)) { __gl_fallback(st, px, bw, bh); return; }
        gGl.bindTexture(GUI_GL_TEXTURE_2D, w->texture);
        if (w->texW != bw || w->texH != bh) {
            gGl.texImage2D(GUI_GL_TEXTURE_2D, 0, (int)GUI_GL_RGBA, bw, bh, 0, GUI_GL_RGBA, GUI_GL_UNSIGNED_BYTE, (const void*)px);
            w->texW = bw;
            w->texH = bh;
        } else if (w->es3) {
            // just the damaged rectangle
            gGl.pixelStorei(GUI_GL_UNPACK_ROW_LENGTH, bw);
            gGl.pixelStorei(GUI_GL_UNPACK_SKIP_PIXELS, x);
            gGl.pixelStorei(GUI_GL_UNPACK_SKIP_ROWS, y);
            gGl.texSubImage2D(GUI_GL_TEXTURE_2D, 0, x, y, rw, rh, GUI_GL_RGBA, GUI_GL_UNSIGNED_BYTE, (const void*)px);
            gGl.pixelStorei(GUI_GL_UNPACK_ROW_LENGTH, 0);
            gGl.pixelStorei(GUI_GL_UNPACK_SKIP_PIXELS, 0);
            gGl.pixelStorei(GUI_GL_UNPACK_SKIP_ROWS, 0);
        } else {
            // ES 2 has no row length: the damaged rows, full width
            const unsigned char* rows = (const unsigned char*)px + (unsigned long)y * (unsigned long)bw * 4UL;
            gGl.texSubImage2D(GUI_GL_TEXTURE_2D, 0, 0, y, bw, rh, GUI_GL_RGBA, GUI_GL_UNSIGNED_BYTE, (const void*)rows);
        }
        // The buffer 1:1 at the surface's top-left (the surface may be a
        // little larger or smaller for a moment around a resize).
        // On Wayland the surface is the wl_egl_window, just resized to the
        // buffer (EGL only reports the new size after the next swap); on
        // X11 it follows the window.
        int sw = bw; int sh = bh;
        if (w->wlEgl == (void*)0) {
            gGl.querySurface(dpy, w->surface, GUI_EGL_WIDTH, &sw);
            gGl.querySurface(dpy, w->surface, GUI_EGL_HEIGHT, &sh);
        }
        if (sw != bw || sh != bh) {
            gGl.viewport(0, 0, sw, sh);
            gGl.clearColor(0.0f, 0.0f, 0.0f, 0.0f);
            gGl.clear(GUI_GL_COLOR_BUFFER_BIT);
        }
        gGl.viewport(0, sh - bh, bw, bh);
        float quad[8];
        quad[0] = -1.0f; quad[1] = -1.0f;
        quad[2] = 1.0f;  quad[3] = -1.0f;
        quad[4] = -1.0f; quad[5] = 1.0f;
        quad[6] = 1.0f;  quad[7] = 1.0f;
        gGl.useProgram(w->program);
        gGl.vertexAttribPointer(0U, 2, GUI_GL_FLOAT, (unsigned char)0, 0, (const void*)&quad[0]);
        gGl.enableVertexAttribArray(0U);
        gGl.drawArrays(GUI_GL_TRIANGLE_STRIP, 0, 4);
        unsigned int ok = 0U;
        int full = (x == 0 && y == 0 && rw == bw && rh == bh && sw == bw && sh == bh) ? 1 : 0;
        if (w->d->swapWithDamage != (EglSwapWithDamage_)0 && !full && sw == bw && sh == bh) {
            int rect[4];
            rect[0] = x; rect[1] = sh - (y + rh); rect[2] = rw; rect[3] = rh;   // origin bottom-left
            ok = w->d->swapWithDamage(dpy, w->surface, &rect[0], 1);
        } else {
            ok = gGl.swapBuffers(dpy, w->surface);
        }
        if (!ok) {
            int err = gGl.getError();
            // EGL_BAD_SURFACE / BAD_NATIVE_WINDOW / CONTEXT_LOST: give up on the GPU
            if (err == 0x300D || err == 0x300B || err == 0x300E) { __gl_fallback(st, px, bw, bh); }
        }
    }
}

?&GuiPresenter gui_gpu_presenter() {
    if (!gGlPresenterInit) {
        gGlPresenter.name = "gl";
        gGlPresenter.attach = __gl_attach;
        gGlPresenter.present = __gl_present;
        gGlPresenter.detach = __gl_detach;
        gGlPresenterInit = 1;
    }
    return &gGlPresenter;
}

} // namespace std
