// SafeC Standard Library — GUI: the Metal presenter for the Cocoa backend
// (see gui_gpu.h).
//
// Like gui_cocoa.sc, everything goes through the Objective-C runtime's C
// API (objc_msgSend cast to each call's real signature — assigned to a
// typed variable first, then called: see std/ml/gpu_mps.sc's note on why).
// Metal.framework is dlopen()ed and MTLCreateSystemDefaultDevice() looked
// up at run time, so nothing is linked and a machine without Metal simply
// keeps the CALayer-contents path.
//
// Per window: the content view becomes layer-hosting with a CAMetalLayer
// (BGRA8Unorm, drawableSize = the buffer, contentsScale = the window's
// scale; the backend's own layer is put back on detach) and a shared-storage
// MTLBuffer holding the frame. present() copies only the damaged rows into
// that buffer, then draws one full-drawable triangle whose fragment shader
// reads the RGBA pixel under it straight from the buffer — the render
// target's BGRA order is the GPU's business, so there is no CPU swizzle —
// and presents the drawable (vsync: the layer's displaySyncEnabled
// default). The shaders are compiled once from source
// (newLibraryWithSource:). Every message sent here takes only pointers,
// integers and one CGSize, so no MTLRegion/MTLOrigin struct is ever passed
// by value. The previous frame's command buffer is waited for before the
// buffer is written again.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>
#include <std/mem.h>

namespace std {

// Opaque Objective-C runtime types: an object ('id'; a Class too), a
// selector (SEL), an autorelease pool token; a dlopen() handle and a
// symbol looked up in one (untyped until cast to its signature).
struct ObjcId_   { char opaque_; };
struct ObjcSel_  { char opaque_; };
struct ObjcPool_ { char opaque_; };
struct MtlLib_   { char opaque_; };
typedef ?fn void() MtlSym_;

extern ObjcId_*  objc_getClass(const char* name);
extern ObjcSel_* sel_registerName(const char* name);
extern ObjcId_*  objc_msgSend(ObjcId_* recv, ObjcSel_* op);
extern ObjcPool_* objc_autoreleasePoolPush();
extern void      objc_autoreleasePoolPop(ObjcPool_* pool);
extern MtlLib_*  dlopen(const char* path, int mode);
extern MtlSym_   dlsym(MtlLib_* handle, const char* name);

struct GuiMtlCGSize { double w; double h; };

// objc_msgSend, one cast per call shape (receiver, selector, arguments).
typedef fn ObjcId_*() MtlCreateDevice_;
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*) MtlMsg0;
typedef fn unsigned char*(ObjcId_*, ObjcSel_*) MtlMsgRetBytes;            // -[MTLBuffer contents]
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, ObjcId_*) MtlMsg1;
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, const char*) MtlMsgStr;          // stringWithUTF8String:
typedef fn void(ObjcId_*, ObjcSel_*) MtlVoid0;
typedef fn void(ObjcId_*, ObjcSel_*, ObjcId_*) MtlVoid1;
typedef fn void(ObjcId_*, ObjcSel_*, unsigned long) MtlVoidULong;
typedef fn void(ObjcId_*, ObjcSel_*, signed char) MtlVoidBool;
typedef fn void(ObjcId_*, ObjcSel_*, double) MtlVoidDouble;
typedef fn void(ObjcId_*, ObjcSel_*, struct GuiMtlCGSize) MtlVoidSize;
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, ObjcId_*, ObjcId_*, ObjcId_**) MtlNewLibrary;  // newLibraryWithSource:options:error:
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, ObjcId_*, ObjcId_**) MtlNewPipeline;           // newRenderPipelineStateWithDescriptor:error:
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, unsigned long, unsigned long) MtlNewBuffer;    // newBufferWithLength:options:
typedef fn ObjcId_*(ObjcId_*, ObjcSel_*, unsigned long) MtlMsgULong;                    // objectAtIndexedSubscript:
typedef fn void(ObjcId_*, ObjcSel_*, ObjcId_*, unsigned long, unsigned long) MtlSetBuffer;   // setFragmentBuffer:offset:atIndex:
typedef fn void(ObjcId_*, ObjcSel_*, const unsigned char*, unsigned long, unsigned long) MtlSetBytes; // setFragmentBytes:length:atIndex:
typedef fn void(ObjcId_*, ObjcSel_*, unsigned long, unsigned long, unsigned long) MtlDraw;  // drawPrimitives:vertexStart:vertexCount:

#define GUI_MTL_BGRA8 80UL           // MTLPixelFormatBGRA8Unorm

static int gMtlLoaded = 0;            // 0 not tried, 1 ok, -1 unavailable
static MtlCreateDevice_ gMtlCreateDevice;
static ObjcId_* gMtlDevice = (ObjcId_*)0;
static ObjcId_* gMtlQueue = (ObjcId_*)0;
static ObjcId_* gMtlPipeline = (ObjcId_*)0;

// A full-drawable triangle; each fragment reads its pixel from the frame
// buffer (RGBA8, premultiplied, row-major, top-left origin like Metal's
// window coordinates) and returns it as is.
static const char* __mtl_shaders() {
    return
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct V { float4 pos [[position]]; };\n"
    "vertex V safec_vs(uint vid [[vertex_id]]) {\n"
    "  float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));\n"
    "  V o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;\n"
    "}\n"
    "fragment float4 safec_fs(V in [[stage_in]], device const uchar4* px [[buffer(0)]],\n"
    "                         constant uint2& size [[buffer(1)]]) {\n"
    "  uint2 p = uint2(in.pos.xy);\n"
    "  if (p.x >= size.x || p.y >= size.y) return float4(0.0);\n"
    "  return float4(px[p.y * size.x + p.x]) / 255.0;\n"
    "}\n";
}

static ObjcSel_* __mtl_sel(const char* name) { unsafe { return sel_registerName(name); } }

static ObjcId_* __mtl_send0(ObjcId_* obj, const char* sel) {
    unsafe {
        MtlMsg0 m = (MtlMsg0)objc_msgSend;
        return m(obj, __mtl_sel(sel));
    }
}

static void __mtl_void0(ObjcId_* obj, const char* sel) {
    unsafe {
        MtlVoid0 m = (MtlVoid0)objc_msgSend;
        m(obj, __mtl_sel(sel));
    }
}

static void __mtl_void1(ObjcId_* obj, const char* sel, ObjcId_* arg) {
    unsafe {
        MtlVoid1 m = (MtlVoid1)objc_msgSend;
        m(obj, __mtl_sel(sel), arg);
    }
}

static void __mtl_bool(ObjcId_* obj, const char* sel, int v) {
    unsafe {
        MtlVoidBool m = (MtlVoidBool)objc_msgSend;
        m(obj, __mtl_sel(sel), v ? (signed char)1 : (signed char)0);
    }
}

static void __mtl_release(ObjcId_* obj) {
    unsafe { if (obj != null) { __mtl_void0(obj, "release"); } }
}

static int __mtl_load() {
    if (gMtlLoaded != 0) { return gMtlLoaded > 0 ? 1 : 0; }
    gMtlLoaded = -1;
    unsafe {
        MtlLib_* lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", 1 /* RTLD_LAZY */);
        if (lib == null) { return 0; }
        dlopen("/System/Library/Frameworks/QuartzCore.framework/QuartzCore", 1);
        match (dlsym(lib, "MTLCreateSystemDefaultDevice")) {
            case some(p): { gMtlCreateDevice = (MtlCreateDevice_)p; }
            case null: { return 0; }
        }
        if (objc_getClass("CAMetalLayer") == null) { return 0; }
        ObjcId_* dev = gMtlCreateDevice();
        if (dev == null) { return 0; }
        ObjcId_* q = __mtl_send0(dev, "newCommandQueue");
        if (q == null) { return 0; }
        // the pipeline: compile the shaders, BGRA8 color attachment
        ObjcPool_* pool = objc_autoreleasePoolPush();
        MtlMsg1 strMsg = (MtlMsg1)objc_msgSend;
        MtlMsgStr nsStr = (MtlMsgStr)objc_msgSend;
        ObjcId_* src = nsStr(objc_getClass("NSString"), __mtl_sel("stringWithUTF8String:"), __mtl_shaders());
        ObjcId_* err = null;
        MtlNewLibrary newLib = (MtlNewLibrary)objc_msgSend;
        ObjcId_* lib2 = newLib(dev, __mtl_sel("newLibraryWithSource:options:error:"), src, null, &err);
        if (lib2 == null) { objc_autoreleasePoolPop(pool); return 0; }
        ObjcId_* vs = strMsg(lib2, __mtl_sel("newFunctionWithName:"), nsStr(objc_getClass("NSString"), __mtl_sel("stringWithUTF8String:"), "safec_vs"));
        ObjcId_* fs = strMsg(lib2, __mtl_sel("newFunctionWithName:"), nsStr(objc_getClass("NSString"), __mtl_sel("stringWithUTF8String:"), "safec_fs"));
        if (vs == null || fs == null) { objc_autoreleasePoolPop(pool); return 0; }
        ObjcId_* desc = __mtl_send0(__mtl_send0(objc_getClass("MTLRenderPipelineDescriptor"), "alloc"), "init");
        __mtl_void1(desc, "setVertexFunction:", vs);
        __mtl_void1(desc, "setFragmentFunction:", fs);
        MtlMsgULong at = (MtlMsgULong)objc_msgSend;
        ObjcId_* att = at(__mtl_send0(desc, "colorAttachments"), __mtl_sel("objectAtIndexedSubscript:"), 0UL);
        MtlVoidULong setU = (MtlVoidULong)objc_msgSend;
        setU(att, __mtl_sel("setPixelFormat:"), GUI_MTL_BGRA8);
        MtlNewPipeline newPipe = (MtlNewPipeline)objc_msgSend;
        ObjcId_* pipe = newPipe(dev, __mtl_sel("newRenderPipelineStateWithDescriptor:error:"), desc, &err);
        __mtl_release(desc);
        __mtl_release(vs);
        __mtl_release(fs);
        __mtl_release(lib2);
        objc_autoreleasePoolPop(pool);
        if (pipe == null) { return 0; }
        gMtlDevice = dev;
        gMtlQueue = q;
        gMtlPipeline = pipe;
    }
    gMtlLoaded = 1;
    return 1;
}

struct GuiMtlWindow {
    ObjcId_* view;
    ObjcId_* oldLayer;       // the backend's layer, retained, put back on detach
    ObjcId_* layer;          // CAMetalLayer, retained
    ObjcId_* buffer;         // MTLBuffer (shared): the frame, RGBA
    unsigned char* data;     // its contents
    int bufW; int bufH;
    ObjcId_* lastCmd;        // the previous frame's command buffer, retained
};

static struct GuiPresenter gMtlPresenter;
static int gMtlPresenterInit = 0;

static void __mtl_free_buffer(struct GuiMtlWindow* w) {
    unsafe {
        __mtl_release(w->buffer);
        w->buffer = null;
        w->data = (unsigned char*)0;
        w->bufW = 0;
        w->bufH = 0;
    }
}

static int __mtl_make_buffer(struct GuiMtlWindow* w, int bw, int bh) {
    unsafe {
        __mtl_free_buffer(w);
        MtlNewBuffer nb = (MtlNewBuffer)objc_msgSend;
        w->buffer = nb(gMtlDevice, __mtl_sel("newBufferWithLength:options:"),
                       (unsigned long)bw * (unsigned long)bh * 4UL, 0UL /* StorageModeShared */);
        if (w->buffer == null) { return 0; }
        MtlMsgRetBytes contents = (MtlMsgRetBytes)objc_msgSend;
        w->data = contents(w->buffer, __mtl_sel("contents"));
        w->bufW = bw;
        w->bufH = bh;
        return 1;
    }
}

static void __mtl_free(struct GuiMtlWindow* w) {
    unsafe {
        if (w->lastCmd != null) {
            __mtl_void0(w->lastCmd, "waitUntilCompleted");
            __mtl_release(w->lastCmd);
        }
        __mtl_free_buffer(w);
        if (w->view != null && w->oldLayer != null) {
            __mtl_void1(w->view, "setLayer:", w->oldLayer);
            __mtl_bool(w->view, "setWantsLayer:", 1);
        }
        __mtl_release(w->oldLayer);
        __mtl_release(w->layer);
        dealloc(w);
    }
}

static int __mtl_attach(&GuiWindowState st, struct GuiNativeHandle h) {
    unsafe {
        if (h.kind != GUI_NATIVE_COCOA || h.view == 0ULL) { return 0; }
        if (!__mtl_load()) { return 0; }
        struct GuiMtlWindow* w = (struct GuiMtlWindow*)alloc_zeroed_raw_(sizeof(struct GuiMtlWindow));
        w->view = (ObjcId_*)(unsigned long)h.view;
        ObjcPool_* pool = objc_autoreleasePoolPush();
        ObjcId_* layer = __mtl_send0(objc_getClass("CAMetalLayer"), "layer");
        if (layer == null) { objc_autoreleasePoolPop(pool); dealloc(w); return 0; }
        __mtl_send0(layer, "retain");
        w->layer = layer;
        __mtl_void1(layer, "setDevice:", gMtlDevice);
        MtlVoidULong setU = (MtlVoidULong)objc_msgSend;
        setU(layer, __mtl_sel("setPixelFormat:"), GUI_MTL_BGRA8);
        __mtl_bool(layer, "setFramebufferOnly:", 0);
        __mtl_bool(layer, "setOpaque:", 0);
        MtlVoidDouble setD = (MtlVoidDouble)objc_msgSend;
        setD(layer, __mtl_sel("setContentsScale:"), st.scale);
        struct GuiMtlCGSize sz;
        sz.w = (double)st.pixelW;
        sz.h = (double)st.pixelH;
        MtlVoidSize setS = (MtlVoidSize)objc_msgSend;
        setS(layer, __mtl_sel("setDrawableSize:"), sz);
        ObjcId_* old = __mtl_send0(w->view, "layer");
        if (old != null) { __mtl_send0(old, "retain"); }
        w->oldLayer = old;
        // layer-hosting: set the layer first, then wantsLayer
        __mtl_void1(w->view, "setLayer:", layer);
        __mtl_bool(w->view, "setWantsLayer:", 1);
        objc_autoreleasePoolPop(pool);
        st.presenterData = (?&Erased)w;
        return 1;
    }
}

static void __mtl_detach(&GuiWindowState st) {
    unsafe {
        if (st.presenterData.is_null()) { return; }
        __mtl_free((struct GuiMtlWindow*)st.presenterData);
        st.presenterData = null;
    }
}

static void __mtl_fallback(&GuiWindowState st, const &unsigned char px, int bw, int bh) {
    __mtl_detach(st);
    st.presenter = null;
    &GuiBackend b = gui_backend();
    b.window_present(st, px, bw, bh, 0, 0, bw, bh);
}

static void __mtl_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                          int x, int y, int rw, int rh) {
    unsafe {
        struct GuiMtlWindow* w = (struct GuiMtlWindow*)st.presenterData;
        if (w == (struct GuiMtlWindow*)0) { return; }
        ObjcPool_* pool = objc_autoreleasePoolPush();
        if (w->lastCmd != null) {
            __mtl_void0(w->lastCmd, "waitUntilCompleted");
            __mtl_release(w->lastCmd);
            w->lastCmd = null;
        }
        if (w->bufW != bw || w->bufH != bh) {
            if (!__mtl_make_buffer(w, bw, bh)) {
                objc_autoreleasePoolPop(pool);
                __mtl_fallback(st, px, bw, bh);
                return;
            }
            y = 0; rh = bh;
            struct GuiMtlCGSize sz;
            sz.w = (double)bw;
            sz.h = (double)bh;
            MtlVoidSize setS = (MtlVoidSize)objc_msgSend;
            setS(w->layer, __mtl_sel("setDrawableSize:"), sz);
            MtlVoidDouble setD = (MtlVoidDouble)objc_msgSend;
            setD(w->layer, __mtl_sel("setContentsScale:"), st.scale);
        }
        // the damaged rows into the GPU-visible copy
        unsigned long rowBytes = (unsigned long)bw * 4UL;
        unsigned long n = (unsigned long)rh * rowBytes;
        safe_memcpy((w->data + (unsigned long)y * rowBytes)[0..n],
                    ((const unsigned char*)px + (unsigned long)y * rowBytes)[0..n]);
        ObjcId_* drawable = __mtl_send0(w->layer, "nextDrawable");
        if (drawable == null) { objc_autoreleasePoolPop(pool); return; }   // none free now: skip the frame
        ObjcId_* target = __mtl_send0(drawable, "texture");
        ObjcId_* cmd = __mtl_send0(gMtlQueue, "commandBuffer");
        if (cmd == null || target == null) {
            objc_autoreleasePoolPop(pool);
            __mtl_fallback(st, px, bw, bh);
            return;
        }
        ObjcId_* pass = __mtl_send0(objc_getClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
        MtlMsgULong at = (MtlMsgULong)objc_msgSend;
        ObjcId_* att = at(__mtl_send0(pass, "colorAttachments"), __mtl_sel("objectAtIndexedSubscript:"), 0UL);
        __mtl_void1(att, "setTexture:", target);
        MtlVoidULong setU = (MtlVoidULong)objc_msgSend;
        setU(att, __mtl_sel("setLoadAction:"), 0UL /* DontCare: every pixel is drawn */);
        setU(att, __mtl_sel("setStoreAction:"), 1UL /* Store */);
        MtlMsg1 enc = (MtlMsg1)objc_msgSend;
        ObjcId_* re = enc(cmd, __mtl_sel("renderCommandEncoderWithDescriptor:"), pass);
        if (re == null) {
            objc_autoreleasePoolPop(pool);
            __mtl_fallback(st, px, bw, bh);
            return;
        }
        __mtl_void1(re, "setRenderPipelineState:", gMtlPipeline);
        MtlSetBuffer setBuf = (MtlSetBuffer)objc_msgSend;
        setBuf(re, __mtl_sel("setFragmentBuffer:offset:atIndex:"), w->buffer, 0UL, 0UL);
        unsigned int size[2];
        size[0] = (unsigned int)bw;
        size[1] = (unsigned int)bh;
        MtlSetBytes setBytes = (MtlSetBytes)objc_msgSend;
        setBytes(re, __mtl_sel("setFragmentBytes:length:atIndex:"), (const unsigned char*)&size[0], 8UL, 1UL);
        MtlDraw draw = (MtlDraw)objc_msgSend;
        draw(re, __mtl_sel("drawPrimitives:vertexStart:vertexCount:"), 3UL /* Triangle */, 0UL, 3UL);
        __mtl_void0(re, "endEncoding");
        __mtl_void1(cmd, "presentDrawable:", drawable);
        __mtl_void0(cmd, "commit");
        __mtl_send0(cmd, "retain");
        w->lastCmd = cmd;
        objc_autoreleasePoolPop(pool);
    }
}

?&GuiPresenter gui_gpu_presenter() {
    if (!gMtlPresenterInit) {
        gMtlPresenter.name = "metal";
        gMtlPresenter.attach = __mtl_attach;
        gMtlPresenter.present = __mtl_present;
        gMtlPresenter.detach = __mtl_detach;
        gMtlPresenterInit = 1;
    }
    return &gMtlPresenter;
}

} // namespace std
