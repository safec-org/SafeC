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
//   windows     windows: "canvas" (default) — each window a <canvas>: the
//               glue's 'canvas' option picks it (an element, a selector, or
//               a function (index, title) => canvas), else the next unused
//               canvas[data-safec-gui] of the page, else a new <canvas>
//               appended to <body>. Its CSS size is the window's logical
//               size and its backing store that times devicePixelRatio
//               (hiDpi), so drawing is sharp; with the 'fit' option (or a
//               data-safec-fit attribute) the canvas keeps the size the
//               page's CSS gives it and follows it (ResizeObserver) —
//               GUI_EVENT_RESIZE, as a window would. windows: "frames" —
//               each window a positioned <div class="safec-gui-window"> in
//               a container, with a title bar (drag it: GUI_EVENT_MOVE,
//               set_position()/pos_x() work; its × : GUI_EVENT_CLOSE), a
//               resize grip honoring the size limits, raised on a click.
//   presenting  putImageData of the damaged rectangle, straight from the
//               module's memory (premultiplied pixels converted to
//               ImageData's straight alpha on the way) — or, with the
//               page's present: "webgl" option or GUI_PRESENT_GPU, WebGL2
//               (else WebGL1): the damaged rectangle uploaded into a
//               texture and drawn as one quad (gui_gpu_web_wasm.sc), back
//               to putImageData if there is no WebGL or the context is lost.
//   input       pointer events (with pointer capture while a button is
//               down), wheel (pixel/line/page deltas to lines); keyboard
//               focus is real DOM focus on a hidden <textarea> per window
//               (a click on the canvas focuses it): KeyboardEvent.code/key
//               to GUI_KEY_* and the typed character, and text that comes
//               without a key — IME composition, virtual keyboards,
//               context-menu pastes — as GUI_EVENT_CHAR.
//   HiDPI       devicePixelRatio, followed when it changes (zoom, another
//               display): GUI_EVENT_SCALE.
//   clipboard   gui_clipboard_set_text() writes navigator.clipboard (and the
//               copy event a Ctrl/Cmd+C shortcut fires); gui_clipboard_get_text()
//               returns the text of the last paste event or of the last
//               copy — the glue holds a Ctrl/Cmd+V keydown back until its
//               paste event arrived, so a paste shortcut sees the pasted
//               text. Reading the system clipboard any other way needs a
//               permission browsers don't grant silently.
//   cursors     CSS cursor on the canvas.
//   a11y        canvases are role="img" labelled with the window title, the
//               hidden input carries it too, and gui_web_announce() speaks
//               through an aria-live region.
//   DOM         gui_web_window_element()/gui_web_mount() trade elements with
//               std/wasm/dom.h, so a page mixes DOM and std/gui windows.
//   clock       performance.now(); timers fire from the frame loop.
// Files: fopen()/fread() read the glue's 'files' option (fetchFiles()), e.g.
// fonts at the paths std/gui looks in; the TrueType stack's embedded
// fallback face is there without any. 'safeguard bundle --target wasm32'
// writes build/<name>.web/: the module, this glue and a page.

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
