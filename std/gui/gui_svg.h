#pragma once
// SafeC Standard Library — GUI: minimal SVG rendering.
//
// A real (not stubbed) but deliberately small subset of SVG: <svg>, <g>,
// <rect>, <circle>, <ellipse>, <line>, <polyline>, <polygon>, and <path>
// with M/L/H/V/C/Z commands (cubic beziers are flattened to line segments
// by subdivision). fill/stroke accept "#rrggbb", "none", and a handful of
// named colors (black/white/red/green/blue/yellow/cyan/magenta/gray/
// transparent). No viewBox/transform/gradient/clip-path/text/use support,
// no CSS — this is a scoped renderer for simple vector icon/shape assets,
// not a browser-grade SVG engine. Parses and rasterizes directly into a
// GuiWindow in one pass (no retained SVG DOM) via GuiWindow's drawing methods
// plus a scanline polygon fill for <polygon>/filled <path>.

#include <std/gui/gui.h>

// The entry point is a GuiWindow method, declared in gui.h and defined in
// gui_svg.sc (there is no retained SVG object to hang it on — the
// document is parsed and drawn in one pass):
//
//   int GuiWindow::draw_svg(int ox, int oy, const &char svgText);
//
// It parses 'svgText' (a complete <svg>...</svg> document) and draws it
// with its top-left at (ox, oy). Returns 1 if at least the <svg> root was
// found and parsed, 0 on malformed/unsupported input (some shapes may
// already have been drawn before a later parse failure — this is a
// direct parse-and-draw pass, not transactional). Include this header (or
// just gui.h) and link gui_svg.sc to use it.
