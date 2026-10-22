#pragma once
// SafeC Standard Library — GUI: the widget toolkit's internal interface
// between its implementation files (gui_widget*.sc). Not API: include
// gui_widget.h instead. Declared in a header (rather than per file) so the
// files agree on one set of signatures; every function here is defined in
// one of the widget sources, which gui_widget.h links.
#include <std/gui/gui_widget.h>

namespace std {

// ── metrics and style (gui_widget.sc) ────────────────────────────────────
float  __w_px(const &GuiUi ui, float logical);          // logical -> pixels
int    __w_ipx(const &GuiUi ui, int logical);
float  __w_font_size(const &GuiUi ui, const &Widget w); // pixels
&GuiTypeface __w_face(const &GuiUi ui, const &Widget w);
float  __w_line_h(const &GuiUi ui, const &Widget w);
float  __w_ascent(const &GuiUi ui, const &Widget w);
float  __w_text_w(const &GuiUi ui, const &Widget w, const &char s);
float  __w_text_wn(const &GuiUi ui, const &Widget w, const &char s, unsigned long n);
// Draws one line with its top at y (x: the left edge), clipped by the canvas.
void   __w_text(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, float x, float y, struct GuiColor col);
void   __w_text_n(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, unsigned long n,
                  float x, float y, struct GuiColor col);
// Draws one line vertically centered in (x, y, w, h), aligned by 'align'.
void   __w_text_in(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, float x, float y, float bw, float bh,
                   int align, struct GuiColor col);
int    __w_pad(const &GuiUi ui, const &Widget w, int side);   // 0 left, 1 top, 2 right, 3 bottom
int    __w_spacing(const &GuiUi ui, const &Widget w);
struct GuiColor __w_fg(const &GuiUi ui, const &Widget w);    // text color (disabled-aware)
struct GuiColor __w_bg_or(const &Widget w, struct GuiColor themeColor);
struct GuiColor __w_border_or(const &Widget w, struct GuiColor themeColor);
// A filled rounded box with a border ring (border width in pixels, may be 0).
void   __w_box(&GuiCanvas c, float x, float y, float w, float h, float r, struct GuiColor fill,
               struct GuiColor border, float bw);
void   __w_ring(&GuiCanvas c, float x, float y, float w, float h, float r, float bw, struct GuiColor col);
void   __w_focus_ring(const &GuiUi ui, &GuiCanvas c, const &Widget w, float x, float y, float bw, float bh, float r);
// A small chevron pointing down (dir 0), right (1), up (2) or left (3).
void   __w_chevron(&GuiCanvas c, float cx, float cy, float size, int dir, struct GuiColor col, float lw);
void   __w_check_mark(&GuiCanvas c, float x, float y, float s, struct GuiColor col, float lw);

// ── callbacks and state ──────────────────────────────────────────────────
void   __w_fire_click(&Widget w);
void   __w_fire_change(&Widget w);
void   __w_fire_activate(&Widget w);
int    __w_focusable(const &Widget w);
int    __w_effectively_enabled(const &Widget w);
void   __w_set_cursor(&GuiUi ui, int cursor);
double __w_clamp_value(const &Widget w, double v);
// Formats the value with the widget's decimals into 'out'.
void   __w_format_value(const &Widget w, double v, &String out);

// ── layout (gui_widget.sc) ───────────────────────────────────────────────
// Natural outer size (padding included, margins not), pixels.
void   __w_measure(&GuiUi ui, &Widget w, &int mw, &int mh);
// Height wanted at a given width (wrapping text); default: natural height.
int    __w_height_for(&GuiUi ui, &Widget w, int width);
// Positions w at (x, y, ww, hh) and lays its subtree out.
void   __w_arrange(&GuiUi ui, &Widget w, int x, int y, int ww, int hh);
// The content box (rect minus padding).
void   __w_content(const &GuiUi ui, const &Widget w, &int cx, &int cy, &int cw, &int ch);
void   __w_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
void   __w_paint_children(&GuiUi ui, &GuiCanvas c, &Widget w);

// ── scrolling (gui_widget_layout.sc) ─────────────────────────────────────
// Scrollbars for a scrolled area (viewport x,y,vw,vh, content w.contentW/H).
void   __w_scroll_clamp(&Widget w, float vw, float vh);
void   __w_scrollbars_paint(const &GuiUi ui, &GuiCanvas c, &Widget w, float x, float y, float vw, float vh);
// Handles wheel and scrollbar dragging; 1 if consumed.
int    __w_scrollbars_event(&GuiUi ui, &Widget w, &GuiEvent ev, float x, float y, float vw, float vh);
void   __w_scroll_into_view(&Widget w, float top, float bottom, float vh);

// ── text editing (gui_widget_text.sc) ────────────────────────────────────
int    __w_edit_key(&GuiUi ui, &Widget w, &GuiEvent ev, int multiline);
void   __w_edit_insert(&Widget w, const &char s, unsigned long n, int multiline);

// ── per-kind hooks (one file per family) ─────────────────────────────────
// basic: label, button, checkbox, toggle, radio, slider, progress,
// spinner, separator, image, canvas, color picker
void   __wk_basic_measure(&GuiUi ui, &Widget w, &int cw, &int ch);
void   __wk_basic_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
int    __wk_basic_event(&GuiUi ui, &Widget w, &GuiEvent ev);
void   __wk_basic_arrange(&GuiUi ui, &Widget w);
// Parses a spinner's text into its value (or restores the text).
void   __w_spinner_commit(&Widget w);
// text: textinput, textarea, spinner's field
void   __wk_text_measure(&GuiUi ui, &Widget w, &int cw, &int ch);
void   __wk_text_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
int    __wk_text_event(&GuiUi ui, &Widget w, &GuiEvent ev);
// lists: combo, list, tree, table, popup list
void   __wk_list_measure(&GuiUi ui, &Widget w, &int cw, &int ch);
void   __wk_list_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
int    __wk_list_event(&GuiUi ui, &Widget w, &GuiEvent ev);
// containers: boxes, grid, stack, scroll, splitter, tabs, group, panel,
// statusbar, dialog
void   __wk_layout_measure(&GuiUi ui, &Widget w, &int cw, &int ch);
void   __wk_layout_arrange(&GuiUi ui, &Widget w);
void   __wk_layout_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
void   __wk_layout_paint_over(&GuiUi ui, &GuiCanvas c, &Widget w);   // after children (scrollbars)
int    __wk_layout_event(&GuiUi ui, &Widget w, &GuiEvent ev);
// 1 if a point inside w is in the area its children show (not in a
// scrollbar, the tab bar, a title...): hit testing descends only then.
int    __wk_layout_child_area(&GuiUi ui, &Widget w, float x, float y);
// menus: menubar, menu, menu item, tooltip
void   __wk_menu_measure(&GuiUi ui, &Widget w, &int cw, &int ch);
void   __wk_menu_arrange(&GuiUi ui, &Widget w);
void   __wk_menu_paint(&GuiUi ui, &GuiCanvas c, &Widget w);
int    __wk_menu_event(&GuiUi ui, &Widget w, &GuiEvent ev);
// Fires the menu item whose shortcut matches the key event in the tree
// (and its context menus); 1 if one did.
int    __wk_menu_shortcut(&GuiUi ui, &Widget root, &GuiEvent ev);
// Opens a menubar entry's / submenu's popup.
void   __wk_menu_open(&GuiUi ui, &Widget menu);

// ── popups (gui_widget.sc) ───────────────────────────────────────────────
void   __w_popup_close_from(&GuiUi ui, &Widget popup);   // closes it and everything above
int    __w_is_popup(const &Widget w);

} // namespace std
