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
// GuiWindow in one pass (no retained SVG DOM) via gui_draw.h's primitives
// plus a scanline polygon fill for <polygon>/filled <path>.

namespace std {

// Parses 'svgText' (a complete <svg>...</svg> document) and draws it with
// its top-left at (ox, oy) in 'win'. Returns 1 if at least the <svg> root
// was found and parsed, 0 on malformed/unsupported input (some shapes may
// already have been drawn before a later parse failure — this is a direct
// parse-and-draw pass, not transactional).
int gui_draw_svg(&GuiWindow win, int ox, int oy, const char* svgText);

} // namespace std
