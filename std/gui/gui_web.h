#pragma once
// SafeC Standard Library — GUI: the WebAssembly <canvas> backend.
//
// On a wasm32/wasm64 target std/gui's platform backend is the browser:
// every GuiWindow is a <canvas> element of the page (or an OffscreenCanvas
// where there is no document), and everything else in std/gui — the
// canvas, text, widgets, GuiApp with its timers and animation frames —
// works unchanged on top of it. The host side is std/gui/gui_web.js, an ES
// module shipped next to this header (see its header comment):
//
//   import { loadSafeCGui } from "./gui_web.js";
//   const app = await loadSafeCGui(fetch("app.wasm"), { entry: "start" });
//
// A browser must not block, so a program doesn't call GuiApp::run(): its
// entry point (an export_name function the glue calls once) opens the
// windows, then hands the app to gui_web_run(), which returns at once; the
// glue then calls GuiApp::pump(0) once per requestAnimationFrame until it
// returns 0 (the last window closed or quit()):
//
//   static struct GuiApp gApp;                 // must outlive start()
//   __attribute__((export_name("start")))
//   int start() {
//       gApp = GuiApp::new();
//       ?&GuiWindow w = gApp.open_window("Hello", 480, 320);
//       ...
//       gui_web_run(&gApp);
//       return 0;
//   }
//
// What maps to what:
//   windows     a <canvas>: the glue's 'canvas' option picks it (an
//               element, a selector, or a function (index, title) =>
//               canvas), else the next unused canvas[data-safec-gui] of the
//               page, else a new <canvas> appended to <body>. Its CSS size
//               is the window's logical size and its backing store that
//               times devicePixelRatio (hiDpi), so drawing is sharp; with
//               the 'fit' option (or a data-safec-fit attribute) the canvas
//               keeps the size the page's CSS gives it and follows it
//               (ResizeObserver) — GUI_EVENT_RESIZE, as a window would.
//               Titles go to the canvas' aria-label (and, with the 'title'
//               option, document.title). Positions don't apply.
//   presenting  putImageData of the damaged rectangle, straight from the
//               module's memory (premultiplied pixels are converted to
//               ImageData's straight alpha on the way; opaque ones are
//               copied as they are).
//   input       pointer events (with pointer capture while a button is
//               down), wheel (pixel/line/page deltas to lines), keyboard
//               (KeyboardEvent.code/key to GUI_KEY_* and the typed
//               character), focus/blur, enter/leave; the canvas is
//               focusable (tabindex 0) and focused on a click.
//   HiDPI       devicePixelRatio, followed when it changes (zoom, another
//               display): GUI_EVENT_SCALE.
//   clipboard   gui_clipboard_set_text() writes navigator.clipboard (allowed
//               from an input event); gui_clipboard_get_text() returns the
//               text of the last paste event or of the last copy — the
//               glue holds a Ctrl/Cmd+V keydown back until its paste event
//               arrived, so a paste shortcut sees the pasted text. Reading
//               the system clipboard any other way needs a permission
//               browsers don't grant silently.
//   clock       performance.now(); timers fire from the frame loop.
// Fonts: the TrueType stack's embedded fallback face is always there; load
// others from bytes you fetch() (gui_text.h).

#include <std/gui/gui.h>
#include <std/gui/gui_app.h>

namespace std {

// Lets the JS glue drive 'app' from requestAnimationFrame: one
// app.pump(0) per frame until it returns 0. Returns at once. 'app' must
// stay alive (a static or heap object) after the caller returns.
void gui_web_run(&GuiApp app);

// Screen readers: says 'text' through the page's aria-live region (e.g. a
// status change your UI only draws).
void gui_web_announce(const &char text);

// std/wasm/dom.h interop (the page loaded the module through safec_dom.js's
// loadSafeC, or passed its glue as the domGlue option): the window's
// element — its frame in "frames" mode, else its <canvas> — as a dom.h
// handle (-1 without the DOM glue), to style it or put DOM elements next
// to it; and moving the window's element into DOM element 'parent'
// (1 = done).
int gui_web_window_element(const &GuiWindow win);
int gui_web_mount(&GuiWindow win, int parent);

} // namespace std
