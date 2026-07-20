// SafeC Standard Library — GUI backend: bare metal (a linear framebuffer,
// as set up by a bootloader — e.g. Multiboot2's framebuffer tag, a VESA
// VBE mode, or a board's fixed display-controller memory map). See gui.h
// for the portable API this implements.
//
// Unlike the OS-native backends (Cocoa/Win32/X11), there is no windowing
// system here — 'gui_create_window()' doesn't open anything, it just
// validates that gui_fb_configure() was called first and allocates the
// same CPU-side RGBA8888 draw buffer every backend uses; gui_present()
// copies (and, if needed, pixel-format-converts) that buffer into the
// real framebuffer memory. There is also no OS input stack — a bare-metal
// target's own keyboard/mouse/touch ISR is responsible for calling
// gui_fb_inject_event() with whatever it decodes; gui_poll_event() just
// drains the small queue that fills. Wire up your own PS/2/USB-HID/etc.
// driver to call gui_fb_inject_event() — this file has no opinion on
// which one you use.
//
// Call gui_fb_configure() exactly once, before gui_create_window(), with
// the real framebuffer's base address/pitch/pixel format — typically
// values your bootloader handed you (e.g. a Multiboot2 framebuffer tag's
// framebuffer_addr/framebuffer_pitch/framebuffer_bpp fields).
#pragma once
#include <std/gui/gui.h>
#include <std/mem.sc>

namespace std {

#define GUI_FB_QUEUE_CAP 32

struct GuiFbConfig {
    void* base;           // linear framebuffer physical/virtual base address
    unsigned long pitch;  // bytes per scanline (may exceed width*bytesPerPixel if padded)
    int width;
    int height;
    int bpp;               // bits per pixel — 32 (xRGB/xBGR) or 16 (RGB565) supported
    int redShift; int greenShift; int blueShift; // bit position of each channel's LSB (32bpp only; ignored for 16bpp RGB565)
};

static struct GuiFbConfig gFbConfig;
static int gFbConfigured = 0;

// Must be called once, before gui_create_window(), with the real
// framebuffer's parameters (see struct GuiFbConfig's field comments).
void gui_fb_configure(struct GuiFbConfig cfg) {
    gFbConfig = cfg;
    gFbConfigured = 1;
}

struct GuiFbPlatform {
    struct GuiEvent queue[GUI_FB_QUEUE_CAP];
    int queueHead;
    int queueTail;
};

// A target's own input driver/ISR calls this with whatever it decodes
// (e.g. a PS/2 scancode translated to a GuiEvent) — gui_poll_event() then
// hands it back out through the normal API, same as every other backend.
void gui_fb_inject_event(&GuiWindow win, struct GuiEvent ev) {
    int hasPlatform = 0;
    unsafe { if (win.platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform) { return; }
    unsafe {
        struct GuiFbPlatform* p = (struct GuiFbPlatform*)win.platform;
        int next = (p->queueTail + 1) % GUI_FB_QUEUE_CAP;
        if (next == p->queueHead) { return; } // full — drop
        p->queue[p->queueTail] = ev;
        p->queueTail = next;
    }
}

struct GuiWindow gui_create_window(const char* title, int width, int height) {
    (void)title; // no window chrome/titlebar on bare metal
    struct GuiWindow win;
    win.platform = (void*)0;
    win.width = width;
    win.height = height;
    win.shouldClose = 0;

    if (!gFbConfigured) {
        unsafe { win.pixels = (&heap unsigned char)0; }
        return win; // caller must check win.platform == NULL, per gui.h
    }

    unsigned long npix = (unsigned long)width * (unsigned long)height * 4UL;
    unsafe {
        win.pixels = (&heap unsigned char)alloc(npix);
        memset((void*)win.pixels, 0, npix);

        struct GuiFbPlatform* p = (struct GuiFbPlatform*)alloc((unsigned long)sizeof(struct GuiFbPlatform));
        p->queueHead = 0;
        p->queueTail = 0;
        win.platform = (void*)p;
    }
    return win;
}

// gui_draw.h always writes RGBA8888 into win.pixels via gui_set_pixel();
// this backend's job is turning that into whatever the real hardware
// framebuffer's pixel format is when presenting.
void gui_set_pixel(&GuiWindow win, int x, int y, unsigned int rgba) {
    int oob = 0;
    if (x < 0 || y < 0 || x >= win.width || y >= win.height) { oob = 1; }
    if (oob) { return; }
    unsafe {
        unsigned long idx = ((unsigned long)y * (unsigned long)win.width + (unsigned long)x) * 4UL;
        win.pixels[idx + 0UL] = (unsigned char)((rgba >> 24) & 0xFFU); // R
        win.pixels[idx + 1UL] = (unsigned char)((rgba >> 16) & 0xFFU); // G
        win.pixels[idx + 2UL] = (unsigned char)((rgba >> 8)  & 0xFFU); // B
        win.pixels[idx + 3UL] = (unsigned char)(rgba & 0xFFU);          // A
    }
}

void gui_present(&GuiWindow win) {
    int hasPlatform = 0;
    unsafe { if (win.platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform || !gFbConfigured) { return; }

    unsafe {
        unsigned char* fb = (unsigned char*)gFbConfig.base;
        int rowLimit = win.height < gFbConfig.height ? win.height : gFbConfig.height;
        int colLimit = win.width < gFbConfig.width ? win.width : gFbConfig.width;
        int y = 0;
        while (y < rowLimit) {
            unsigned long srcRow = (unsigned long)y * (unsigned long)win.width * 4UL;
            unsigned long dstRow = (unsigned long)y * gFbConfig.pitch;
            int x = 0;
            while (x < colLimit) {
                unsigned long si = srcRow + (unsigned long)x * 4UL;
                unsigned char r = win.pixels[si + 0UL];
                unsigned char g = win.pixels[si + 1UL];
                unsigned char b = win.pixels[si + 2UL];
                if (gFbConfig.bpp == 32) {
                    unsigned int packedColor = ((unsigned int)r << gFbConfig.redShift) |
                                           ((unsigned int)g << gFbConfig.greenShift) |
                                           ((unsigned int)b << gFbConfig.blueShift);
                    unsigned long di = dstRow + (unsigned long)x * 4UL;
                    fb[di + 0UL] = (unsigned char)(packedColor & 0xFFU);
                    fb[di + 1UL] = (unsigned char)((packedColor >> 8) & 0xFFU);
                    fb[di + 2UL] = (unsigned char)((packedColor >> 16) & 0xFFU);
                    fb[di + 3UL] = (unsigned char)((packedColor >> 24) & 0xFFU);
                } else if (gFbConfig.bpp == 16) {
                    // RGB565.
                    unsigned int r5 = ((unsigned int)r >> 3) & 0x1FU;
                    unsigned int g6 = ((unsigned int)g >> 2) & 0x3FU;
                    unsigned int b5 = ((unsigned int)b >> 3) & 0x1FU;
                    unsigned int packed16 = (r5 << 11) | (g6 << 5) | b5;
                    unsigned long di = dstRow + (unsigned long)x * 2UL;
                    fb[di + 0UL] = (unsigned char)(packed16 & 0xFFU);
                    fb[di + 1UL] = (unsigned char)((packed16 >> 8) & 0xFFU);
                }
                x = x + 1;
            }
            y = y + 1;
        }
    }
}

int gui_poll_event(&GuiWindow win, &GuiEvent outEvent) {
    outEvent.kind = GUI_EVENT_NONE;
    outEvent.defaultPrevented = 0;
    int hasPlatform = 0;
    unsafe { if (win.platform != (void*)0) { hasPlatform = 1; } }
    if (!hasPlatform) { return 0; }
    unsafe {
        struct GuiFbPlatform* p = (struct GuiFbPlatform*)win.platform;
        if (p->queueHead == p->queueTail) { return 0; }
        struct GuiEvent* dst = (struct GuiEvent*)outEvent;
        *dst = p->queue[p->queueHead];
        p->queueHead = (p->queueHead + 1) % GUI_FB_QUEUE_CAP;
        return 1;
    }
}

void gui_destroy_window(&GuiWindow win) {
    unsafe {
        if (win.platform != (void*)0) { dealloc(win.platform); }
        if ((void*)win.pixels != (void*)0) { dealloc((void*)win.pixels); }
        win.platform = (void*)0;
    }
}

} // namespace std
