#pragma once
// SafeC Standard Library — GUI: GPU-accelerated presentation.
//
// std/gui renders on the CPU (identical pixels everywhere); this is only
// about the last step, getting a finished frame onto the screen. By default
// each backend blits it in software (XPutImage, wl_shm, StretchDIBits, a
// CALayer's contents). A window can instead present through the GPU: the
// buffer is uploaded into a texture — only the rows/rectangle that
// changed — and drawn as one quad by the graphics API of the platform:
//
//   "gl"     OpenGL ES 2 through EGL, on X11 and Wayland (libEGL and
//            libGLESv2 are dlopen()ed; Mesa, NVIDIA, ...). vsync on X11
//            (swap interval 1); on Wayland the compositor paces frames
//            and nothing ever tears. Damage is passed on with
//            EGL_KHR/EXT_swap_buffers_with_damage when available.
//   "metal"  Metal on macOS: a CAMetalLayer on the window's view, the frame
//            copied into an MTLTexture (replaceRegion) and blitted into the
//            layer's drawable (Metal.framework is loaded at run time).
//   "d3d11"  Direct3D 11 on Windows: a flip-model DXGI swap chain on the
//            HWND (FLIP_DISCARD, else FLIP_SEQUENTIAL), the frame copied
//            into a texture with UpdateSubresource and to the back buffer
//            with CopyResource, presented with vsync (d3d11.dll is loaded
//            at run time).
// Use it per window (GuiWindowOptions.present = GUI_PRESENT_GPU, or
// gui_window_use_gpu() on a live window) or for every window of a program
// by running it with SAFEC_GUI_PRESENT=gpu (SAFEC_GUI_PRESENT=cpu forces
// the software path). Any failure — no driver, no library, a context or
// swap chain that can't be made, a device lost later — falls back to the
// backend's software presentation automatically, for good, for that
// window; gui_window_uses_gpu() tells which path a window is on.
//
// Worth it for large windows redrawn often (animation, video, games) where
// the software blit dominates; for mostly static UIs the CPU path is just
// as good. The pixels shown are the same either way.

#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>

namespace std {

// The GPU presenter of this platform and backend ("gl", "metal", "d3d11"),
// or null where there is none (WebAssembly, bare metal, headless).
?&GuiPresenter gui_gpu_presenter();
// Attaches it to 'win'. 1 if the window now presents through the GPU, 0 if
// it doesn't (no presenter, or it failed: the CPU path stays).
int gui_window_use_gpu(&GuiWindow win);
// Back to the backend's software presentation.
void gui_window_use_cpu(&GuiWindow win);
// 1 while 'win' presents through a GPU presenter.
int gui_window_uses_gpu(const &GuiWindow win);
// The presenter's name ("gl", "metal", "d3d11"), or "cpu".
const &char gui_window_presenter_name(const &GuiWindow win);

// For GuiWindow::create_with: whether a window with this
// GuiWindowOptions.present should get the GPU presenter
// ($SAFEC_GUI_PRESENT decides GUI_PRESENT_DEFAULT).
int gui_gpu_wanted(int present);

} // namespace std
