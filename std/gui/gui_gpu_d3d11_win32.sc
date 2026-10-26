// SafeC Standard Library — GUI: the Direct3D 11 presenter for the Win32
// backend (see gui_gpu.h).
//
// d3d11.dll is loaded with LoadLibraryA and D3D11CreateDevice looked up at
// run time (a hardware device, else WARP), so nothing is linked. COM is
// called through explicit vtables, as std/audio's WASAPI backend does
// (index = method order in the interface, IUnknown's three first; checked
// against the Windows SDK / MinGW-w64 <d3d11.h>, <dxgi.h>, <dxgi1_2.h>).
//
// Per window: a flip-model DXGI swap chain on the HWND (two R8G8B8A8_UNORM
// buffers — our byte order, no swizzle — FLIP_DISCARD, else
// FLIP_SEQUENTIAL on Windows 8, DXGI_SCALING_NONE so the buffer shows 1:1
// at the top-left even while the window is being resized) and one
// DEFAULT-usage texture holding the frame. present() uploads only the
// damaged rectangle with UpdateSubresource, copies the texture into the
// back buffer with CopyResource and presents with vsync (sync interval 1)
// through IDXGISwapChain1::Present1, passing the damaged rectangle as the
// dirty rect. A size change resizes the buffers (ResizeBuffers) and the
// texture; a removed or reset device sends the window back to GDI.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>
#include <std/mem.h>

namespace std {

// A loaded DLL (HMODULE), and a function looked up in one (FARPROC):
// untyped until cast to its real signature (in 'unsafe').
struct D3dLib_ { char opaque_; };
typedef ?fn void() D3dSym_;
extern D3dLib_* LoadLibraryA(const char* name);
extern D3dSym_ GetProcAddress(D3dLib_* module, const char* name);

// A COM interface pointer (ID3D11Device, IDXGISwapChain1, ...): an object
// whose first field is its method table (IUnknown's three methods, then
// the interface's, in order). Each method is cast to its signature below.
typedef fn void() D3dMethod_;
struct D3dCom_ { const D3dMethod_* vtbl; };
// An HWND, and the structs passed only as null (D3D11_SUBRESOURCE_DATA,
// DXGI_SWAP_CHAIN_FULLSCREEN_DESC).
struct D3dHwnd_ { char opaque_; };
struct D3dSubresourceData_ { char opaque_; };
struct D3dFullscreenDesc_ { char opaque_; };

struct GuiD3dGuid {
    unsigned int d1;
    unsigned short d2;
    unsigned short d3;
    unsigned char d4[8];
};

static struct GuiD3dGuid __d3d_guid(unsigned int d1, unsigned short d2, unsigned short d3,
                                    unsigned char a, unsigned char b, unsigned char c, unsigned char d,
                                    unsigned char e, unsigned char f, unsigned char g, unsigned char h) {
    struct GuiD3dGuid r;
    r.d1 = d1; r.d2 = d2; r.d3 = d3;
    r.d4[0] = a; r.d4[1] = b; r.d4[2] = c; r.d4[3] = d;
    r.d4[4] = e; r.d4[5] = f; r.d4[6] = g; r.d4[7] = h;
    return r;
}

// {54EC77FA-1377-44E6-8C32-88FD5F44C84C}
static struct GuiD3dGuid __d3d_iid_dxgi_device() {
    return __d3d_guid(0x54EC77FAU, (unsigned short)0x1377, (unsigned short)0x44E6, (unsigned char)0x8C, (unsigned char)0x32,
                      (unsigned char)0x88, (unsigned char)0xFD, (unsigned char)0x5F, (unsigned char)0x44, (unsigned char)0xC8, (unsigned char)0x4C);
}
// {50C83A1C-E072-4C48-87B0-3630FA36A6D0}
static struct GuiD3dGuid __d3d_iid_factory2() {
    return __d3d_guid(0x50C83A1CU, (unsigned short)0xE072, (unsigned short)0x4C48, (unsigned char)0x87, (unsigned char)0xB0,
                      (unsigned char)0x36, (unsigned char)0x30, (unsigned char)0xFA, (unsigned char)0x36, (unsigned char)0xA6, (unsigned char)0xD0);
}
// {6F15AAF2-D208-4E89-9AB4-489535D34F9C}
static struct GuiD3dGuid __d3d_iid_texture2d() {
    return __d3d_guid(0x6F15AAF2U, (unsigned short)0xD208, (unsigned short)0x4E89, (unsigned char)0x9A, (unsigned char)0xB4,
                      (unsigned char)0x48, (unsigned char)0x95, (unsigned char)0x35, (unsigned char)0xD3, (unsigned char)0x4F, (unsigned char)0x9C);
}

// DXGI_SWAP_CHAIN_DESC1
struct GuiD3dSwapDesc {
    unsigned int width;
    unsigned int height;
    unsigned int format;
    int stereo;
    unsigned int sampleCount;
    unsigned int sampleQuality;
    unsigned int bufferUsage;
    unsigned int bufferCount;
    unsigned int scaling;
    unsigned int swapEffect;
    unsigned int alphaMode;
    unsigned int flags;
};

// D3D11_TEXTURE2D_DESC
struct GuiD3dTexDesc {
    unsigned int width;
    unsigned int height;
    unsigned int mipLevels;
    unsigned int arraySize;
    unsigned int format;
    unsigned int sampleCount;
    unsigned int sampleQuality;
    unsigned int usage;
    unsigned int bindFlags;
    unsigned int cpuAccessFlags;
    unsigned int miscFlags;
};

// D3D11_BOX
struct GuiD3dBox {
    unsigned int left; unsigned int top; unsigned int front;
    unsigned int right; unsigned int bottom; unsigned int back;
};

// RECT, DXGI_PRESENT_PARAMETERS
struct GuiD3dRect { int left; int top; int right; int bottom; };
struct GuiD3dPoint { int x; int y; };
struct GuiD3dPresentParams {
    unsigned int dirtyRectsCount;
    struct GuiD3dRect* dirtyRects;
    struct GuiD3dRect* scrollRect;
    struct GuiD3dPoint* scrollOffset;
};

#define GUI_DXGI_FORMAT_R8G8B8A8_UNORM 28U
#define GUI_DXGI_USAGE_RENDER_TARGET_OUTPUT 0x20U
#define GUI_DXGI_SCALING_STRETCH 0U
#define GUI_DXGI_SCALING_NONE 1U
#define GUI_DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL 3U
#define GUI_DXGI_SWAP_EFFECT_FLIP_DISCARD 4U
#define GUI_DXGI_MWA_NO_ALT_ENTER 2U

typedef fn int(D3dCom_*, int, D3dLib_*, unsigned int, const unsigned int*, unsigned int, unsigned int,
               D3dCom_**, unsigned int*, D3dCom_**) D3d11CreateDevice_;
typedef fn unsigned int(D3dCom_*) D3dRelease_;                                     // IUnknown 2
typedef fn int(D3dCom_*, const GuiD3dGuid*, D3dCom_**) D3dQueryInterface_;         // IUnknown 0, IDXGIObject::GetParent 6
typedef fn int(D3dCom_*, D3dCom_**) D3dGetAdapter_;                                // IDXGIDevice 7
typedef fn int(D3dCom_*, D3dCom_*, D3dHwnd_*, const GuiD3dSwapDesc*, const D3dFullscreenDesc_*, D3dCom_*, D3dCom_**) D3dCreateSwapChainForHwnd_; // IDXGIFactory2 15
typedef fn int(D3dCom_*, D3dHwnd_*, unsigned int) D3dMakeWindowAssociation_;      // IDXGIFactory 8
typedef fn int(D3dCom_*, unsigned int, const GuiD3dGuid*, D3dCom_**) D3dGetBuffer_; // IDXGISwapChain 9
typedef fn int(D3dCom_*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) D3dResizeBuffers_; // 13
typedef fn int(D3dCom_*, unsigned int, unsigned int) D3dPresent_;                  // IDXGISwapChain 8
typedef fn int(D3dCom_*, unsigned int, unsigned int, const GuiD3dPresentParams*) D3dPresent1_; // IDXGISwapChain1 22
typedef fn int(D3dCom_*, const GuiD3dTexDesc*, const D3dSubresourceData_*, D3dCom_**) D3dCreateTexture2D_; // ID3D11Device 5
typedef fn void(D3dCom_*, D3dCom_*, D3dCom_*) D3dCopyResource_;                    // ID3D11DeviceContext 47
typedef fn void(D3dCom_*, D3dCom_*, unsigned int, const GuiD3dBox*, const unsigned char*, unsigned int, unsigned int) D3dUpdateSubresource_; // 48

// Method 'index' of a COM object's table (the caller casts it to its signature).
static D3dMethod_ __d3d_method(D3dCom_* obj, int index) {
    unsafe { return obj->vtbl[index]; }
}

static void __d3d_release(D3dCom_* obj) {
    unsafe {
        if (obj == (D3dCom_*)0) { return; }
        D3dRelease_ f = (D3dRelease_)__d3d_method(obj, 2);
        f(obj);
    }
}

static int gD3dLoaded = 0;     // 0 not tried, 1 ok, -1 unavailable
static D3d11CreateDevice_ gD3dCreateDevice;

static int __d3d_load() {
    if (gD3dLoaded != 0) { return gD3dLoaded > 0 ? 1 : 0; }
    gD3dLoaded = -1;
    unsafe {
        D3dLib_* lib = LoadLibraryA("d3d11.dll");
        if (lib == (D3dLib_*)0) { return 0; }
        match (GetProcAddress(lib, "D3D11CreateDevice")) {
            case some(p): { gD3dCreateDevice = (D3d11CreateDevice_)p; }
            case null: { return 0; }
        }
    }
    gD3dLoaded = 1;
    return 1;
}

struct GuiD3dWindow {
    D3dCom_* device;       // ID3D11Device
    D3dCom_* context;      // ID3D11DeviceContext
    D3dCom_* swapChain;    // IDXGISwapChain1
    D3dCom_* texture;      // ID3D11Texture2D, the frame
    int texW; int texH;
    int scW; int scH;   // swap chain buffer size
    int flipDiscard;
};

static struct GuiPresenter gD3dPresenter;
static int gD3dPresenterInit = 0;

static void __d3d_free(struct GuiD3dWindow* w) {
    unsafe {
        __d3d_release(w->texture);
        __d3d_release(w->swapChain);
        __d3d_release(w->context);
        __d3d_release(w->device);
        dealloc(w);
    }
}

static int __d3d_attach(&GuiWindowState st, struct GuiNativeHandle h) {
    unsafe {
        if (h.kind != GUI_NATIVE_WIN32 || h.window == 0ULL) { return 0; }
        if (!__d3d_load()) { return 0; }
        D3dHwnd_* hwnd = (D3dHwnd_*)(unsigned long long)h.window;
        struct GuiD3dWindow* w = (struct GuiD3dWindow*)alloc_zeroed_raw_(sizeof(struct GuiD3dWindow));
        unsigned int level = 0U;
        // HARDWARE (1), else WARP (5); BGRA_SUPPORT (0x20) for DXGI interop.
        int hr = gD3dCreateDevice((D3dCom_*)0, 1, (D3dLib_*)0, 0x20U, (const unsigned int*)0, 0U, 7U /* SDK_VERSION */,
                                  &w->device, &level, &w->context);
        if (hr < 0) {
            hr = gD3dCreateDevice((D3dCom_*)0, 5, (D3dLib_*)0, 0x20U, (const unsigned int*)0, 0U, 7U,
                                  &w->device, &level, &w->context);
        }
        if (hr < 0 || w->device == (D3dCom_*)0) { __d3d_free(w); return 0; }
        // device -> IDXGIDevice -> adapter -> IDXGIFactory2
        struct GuiD3dGuid iidDev = __d3d_iid_dxgi_device();
        D3dCom_* dxgiDev = (D3dCom_*)0;
        D3dQueryInterface_ qi = (D3dQueryInterface_)__d3d_method(w->device, 0);
        if (qi(w->device, &iidDev, &dxgiDev) < 0) { __d3d_free(w); return 0; }
        D3dCom_* adapter = (D3dCom_*)0;
        D3dGetAdapter_ getAdapter = (D3dGetAdapter_)__d3d_method(dxgiDev, 7);
        hr = getAdapter(dxgiDev, &adapter);
        __d3d_release(dxgiDev);
        if (hr < 0) { __d3d_free(w); return 0; }
        struct GuiD3dGuid iidFac = __d3d_iid_factory2();
        D3dCom_* factory = (D3dCom_*)0;
        D3dQueryInterface_ getParent = (D3dQueryInterface_)__d3d_method(adapter, 6);
        hr = getParent(adapter, &iidFac, &factory);
        __d3d_release(adapter);
        if (hr < 0) { __d3d_free(w); return 0; }   // no IDXGIFactory2: Windows 7

        struct GuiD3dSwapDesc d;
        safe_memset_obj(&d, 0);
        d.width = (unsigned int)(st.pixelW > 0 ? st.pixelW : 1);
        d.height = (unsigned int)(st.pixelH > 0 ? st.pixelH : 1);
        d.format = GUI_DXGI_FORMAT_R8G8B8A8_UNORM;
        d.sampleCount = 1U;
        d.bufferUsage = GUI_DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.bufferCount = 2U;
        d.scaling = GUI_DXGI_SCALING_NONE;
        d.swapEffect = GUI_DXGI_SWAP_EFFECT_FLIP_DISCARD;
        D3dCreateSwapChainForHwnd_ create = (D3dCreateSwapChainForHwnd_)__d3d_method(factory, 15);
        hr = create(factory, w->device, hwnd, &d, (const D3dFullscreenDesc_*)0, (D3dCom_*)0, &w->swapChain);
        w->flipDiscard = 1;
        if (hr < 0) {   // Windows 8: no FLIP_DISCARD
            d.swapEffect = GUI_DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
            w->flipDiscard = 0;
            hr = create(factory, w->device, hwnd, &d, (const D3dFullscreenDesc_*)0, (D3dCom_*)0, &w->swapChain);
        }
        if (hr < 0) {   // a driver without SCALING_NONE
            d.scaling = GUI_DXGI_SCALING_STRETCH;
            hr = create(factory, w->device, hwnd, &d, (const D3dFullscreenDesc_*)0, (D3dCom_*)0, &w->swapChain);
        }
        if (hr >= 0) {
            D3dMakeWindowAssociation_ mwa = (D3dMakeWindowAssociation_)__d3d_method(factory, 8);
            mwa(factory, hwnd, GUI_DXGI_MWA_NO_ALT_ENTER);
        }
        __d3d_release(factory);
        if (hr < 0 || w->swapChain == (D3dCom_*)0) { __d3d_free(w); return 0; }
        w->scW = (int)d.width;
        w->scH = (int)d.height;
        st.presenterData = (?&Erased)w;
        return 1;
    }
}

static void __d3d_detach(&GuiWindowState st) {
    unsafe {
        if (st.presenterData.is_null()) { return; }
        __d3d_free((struct GuiD3dWindow*)st.presenterData);
        st.presenterData = null;
    }
}

static void __d3d_fallback(&GuiWindowState st, const &unsigned char px, int bw, int bh) {
    __d3d_detach(st);
    st.presenter = null;
    &GuiBackend b = gui_backend();
    b.window_present(st, px, bw, bh, 0, 0, bw, bh);
}

static void __d3d_present(&GuiWindowState st, const &unsigned char px, int bw, int bh,
                          int x, int y, int rw, int rh) {
    unsafe {
        struct GuiD3dWindow* w = (struct GuiD3dWindow*)st.presenterData;
        if (w == (struct GuiD3dWindow*)0) { return; }
        if (w->scW != bw || w->scH != bh) {
            D3dResizeBuffers_ resize = (D3dResizeBuffers_)__d3d_method(w->swapChain, 13);
            if (resize(w->swapChain, 0U, (unsigned int)bw, (unsigned int)bh, 0U, 0U) < 0) {
                __d3d_fallback(st, px, bw, bh);
                return;
            }
            w->scW = bw;
            w->scH = bh;
        }
        if (w->texW != bw || w->texH != bh) {
            __d3d_release(w->texture);
            w->texture = (D3dCom_*)0;
            struct GuiD3dTexDesc td;
            safe_memset_obj(&td, 0);
            td.width = (unsigned int)bw;
            td.height = (unsigned int)bh;
            td.mipLevels = 1U;
            td.arraySize = 1U;
            td.format = GUI_DXGI_FORMAT_R8G8B8A8_UNORM;
            td.sampleCount = 1U;
            td.usage = 0U;   // DEFAULT: a copy source the GPU keeps
            D3dCreateTexture2D_ mk = (D3dCreateTexture2D_)__d3d_method(w->device, 5);
            if (mk(w->device, &td, (const D3dSubresourceData_*)0, &w->texture) < 0 || w->texture == (D3dCom_*)0) {
                __d3d_fallback(st, px, bw, bh);
                return;
            }
            w->texW = bw;
            w->texH = bh;
            x = 0; y = 0; rw = bw; rh = bh;
        }
        struct GuiD3dBox box;
        box.left = (unsigned int)x; box.top = (unsigned int)y; box.front = 0U;
        box.right = (unsigned int)(x + rw); box.bottom = (unsigned int)(y + rh); box.back = 1U;
        const unsigned char* src = (const unsigned char*)px + ((unsigned long long)y * (unsigned long long)bw + (unsigned long long)x) * 4ULL;
        D3dUpdateSubresource_ upd = (D3dUpdateSubresource_)__d3d_method(w->context, 48);
        upd(w->context, w->texture, 0U, &box, src, (unsigned int)bw * 4U, 0U);
        struct GuiD3dGuid iidTex = __d3d_iid_texture2d();
        D3dCom_* back = (D3dCom_*)0;
        D3dGetBuffer_ getBuffer = (D3dGetBuffer_)__d3d_method(w->swapChain, 9);
        if (getBuffer(w->swapChain, 0U, &iidTex, &back) < 0 || back == (D3dCom_*)0) {
            __d3d_fallback(st, px, bw, bh);
            return;
        }
        D3dCopyResource_ copy = (D3dCopyResource_)__d3d_method(w->context, 47);
        copy(w->context, back, w->texture);
        __d3d_release(back);
        struct GuiD3dRect dirty;
        dirty.left = x; dirty.top = y; dirty.right = x + rw; dirty.bottom = y + rh;
        struct GuiD3dPresentParams pp;
        safe_memset_obj(&pp, 0);
        if (!(x == 0 && y == 0 && rw == bw && rh == bh)) {
            pp.dirtyRectsCount = 1U;
            pp.dirtyRects = (struct GuiD3dRect*)&dirty;
        }
        D3dPresent1_ present1 = (D3dPresent1_)__d3d_method(w->swapChain, 22);
        int hr = present1(w->swapChain, 1U, 0U, &pp);
        // DXGI_ERROR_DEVICE_REMOVED / DEVICE_RESET / DEVICE_HUNG: back to GDI
        if ((unsigned int)hr == 0x887A0005U || (unsigned int)hr == 0x887A0007U || (unsigned int)hr == 0x887A0006U) {
            __d3d_fallback(st, px, bw, bh);
        }
    }
}

?&GuiPresenter gui_gpu_presenter() {
    if (!gD3dPresenterInit) {
        gD3dPresenter.name = "d3d11";
        gD3dPresenter.attach = __d3d_attach;
        gD3dPresenter.present = __d3d_present;
        gD3dPresenter.detach = __d3d_detach;
        gD3dPresenterInit = 1;
    }
    return &gD3dPresenter;
}

} // namespace std
