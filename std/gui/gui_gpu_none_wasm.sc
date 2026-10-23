// SafeC Standard Library — GUI: no GPU presenter on this target
// (WebAssembly: the page's <canvas> is the presentation). See gui_gpu.h.
#pragma once
#include <std/gui/gui.h>
#include <std/gui/gui_backend.h>
#include <std/gui/gui_gpu.h>

namespace std {

?&GuiPresenter gui_gpu_presenter() { return null; }

} // namespace std
