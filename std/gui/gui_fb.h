#pragma once
// SafeC Standard Library — GUI: bare-metal framebuffer backend (gui_fb.sc).
//
// The setup calls only this backend has: describe the bootloader-provided
// linear framebuffer once, and feed input events from your own drivers.
// Everything else is the portable API in gui.h.
#include <std/gui/gui.h>

namespace std {

#define GUI_FB_QUEUE_CAP 32   // injected events waiting for GuiWindow::poll_event()

struct GuiFbConfig {
    &volatile unsigned char base;  // linear framebuffer memory (physical/virtual base address)
    unsigned long pitch;  // bytes per scanline (may exceed width*bytesPerPixel if padded)
    int width;
    int height;
    int bpp;               // bits per pixel — 32 (xRGB/xBGR) or 16 (RGB565) supported
    int redShift; int greenShift; int blueShift; // bit position of each channel's LSB (32bpp only; ignored for 16bpp RGB565)
};

// Must be called once, before GuiWindow::create(), with the real
// framebuffer's parameters (see struct GuiFbConfig's field comments).
// Without it GuiWindow::create() fails (its 'platform' is null).
void gui_fb_configure(struct GuiFbConfig cfg);

// A monotonic clock for timers and animation (GuiApp, gui_time_ms()):
// e.g. a SysTick or TSC-based counter, in nanoseconds. Without one, time
// is virtual and only moves while the event loop waits.
void gui_fb_set_clock(fn long long() nowNs);

// Called by a target's own input driver/ISR with whatever it decodes (e.g.
// a PS/2 scancode translated to a GuiEvent); GuiWindow::poll_event() hands it back
// out. Dropped when GUI_FB_QUEUE_CAP events are already waiting.
void gui_fb_inject_event(&GuiWindow win, struct GuiEvent ev);

} // namespace std
