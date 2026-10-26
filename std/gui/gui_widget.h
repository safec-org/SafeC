#pragma once
#include <std/erased.h>
// SafeC Standard Library — GUI: the retained-mode widget toolkit.
//
// A tree of Widgets — containers (vertical/horizontal boxes, grids,
// overlapping stacks, scroll views, splitters, tabs, group boxes, panels)
// and controls (labels, buttons, toggles, check boxes, radio groups,
// sliders, progress bars, number spinners, text inputs, multi-line text
// areas, combo boxes, list / tree / table views, menu bars and context
// menus, tooltips, modal dialogs and message boxes, images, a vector
// canvas, a color picker, separators, status bars) — laid out, painted
// with the anti-aliased canvas (gui_canvas.h) in a light or dark theme,
// and driven by GuiEvents with keyboard focus, Tab navigation, hover /
// pressed / disabled states and the clipboard.
//
//   &Widget root = Widget::vstack();
//   &Widget name = Widget::textinput("Your name");
//   root.add_child(Widget::label("Hello!"));
//   root.add_child(name);
//   root.add_child(Widget::button("OK").on_click(ok_clicked, (?&State)&state));
//
// Driving it: with gui_app.h's GuiApp (app.set_root(win, root) — events,
// layout, painting, timers and tooltips are then automatic, for any number
// of windows), or by hand, once per frame:
//
//   while (win.poll_event(&ev)) { root.dispatch_event(&ev); }
//   root.render(&win);          // layout + paint into win.pixels
//   win.present();
//
// The by-hand calls work on the root widget (as before this toolkit
// existed: root.layout(x, y, w, h), root.render(&win),
// root.dispatch_event(&ev)) and use a GuiUi created for it on first use;
// GuiUi::new(root) gives you that object to hold (themes, focus, popups,
// dialogs, tooltips, scale).
//
// Units: sizes given to widgets (set_pref_size, set_min_size, margins,
// padding, fonts) are logical; the GuiUi multiplies them by the window's
// scale factor, so a hiDpi window gets the same layout at twice the
// pixels. Widget rects (x, y, w, h) are in window pixels.
//
// Layout model: each widget has a natural size (its content, plus
// padding; set_pref_size() overrides either dimension; min/max sizes clamp
// it) and margins. Boxes place children in a row or column with 'spacing'
// between them, give extra space to children with a stretch weight
// (set_stretch, in proportion), and shrink stretching children (down to
// their minimum) when space is short; on the cross axis each child is
// aligned by set_alignment() (START/CENTER/END/STRETCH). Grids size
// columns and rows to their widest/tallest cell, with stretch weights per
// column/row (set_column_stretch / set_row_stretch) and cells spanning
// several. Stacks overlap children, each aligned on both axes. Scroll views
// give their child its natural size and scroll the rest into view.
//
// Callbacks receive the widget and a typed user-data reference (stored
// erased, see the note on Widget's callback fields):
//   on_click(cb, data)     buttons, menu items, toggles/check boxes (after the change), links
//   on_change(cb, data)    any value change: text, check state, slider/spinner value,
//                          selection, color, splitter position, tab
//   on_activate(cb, data)  double click / Enter on a list, tree or table row
//   on_draw(cb, data)      Widget::canvas: draw with the vector canvas
//
// Keyboard: Tab / Shift+Tab move focus through the focusable widgets in
// tree order (skipping hidden and disabled ones, staying inside an open
// modal dialog); Space/Enter activate buttons, toggles and check boxes;
// arrows move sliders, spinners, radio groups, list/tree/table selections
// and tabs; Escape closes a popup, a dialog, or clears focus; menu item
// shortcuts (set_shortcut("Ctrl+S"), or the primary modifier on macOS)
// fire from anywhere in the window. Text fields support selection with
// Shift+arrows/Home/End and the mouse (double click selects a word, triple
// click everything), word-wise moves with Ctrl (Alt on macOS), and
// Cut/Copy/Paste/Select-all through the system clipboard.

#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_font.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_png.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

// ── widget kinds ─────────────────────────────────────────────────────────
#define WIDGET_VSTACK     0
#define WIDGET_HSTACK     1
#define WIDGET_BUTTON     2
#define WIDGET_LABEL      3
#define WIDGET_CHECKBOX   4
#define WIDGET_TEXTINPUT  5
#define WIDGET_SPACER     6
#define WIDGET_SLIDER     7
#define WIDGET_CUSTOM     8
#define WIDGET_TOGGLE     9
#define WIDGET_RADIO      10   // a radio group: one choice of several items
#define WIDGET_PROGRESS   11
#define WIDGET_SPINNER    12   // number input with up/down buttons
#define WIDGET_TEXTAREA   13
#define WIDGET_COMBO      14
#define WIDGET_LIST       15
#define WIDGET_TREE       16
#define WIDGET_TABLE      17
#define WIDGET_TABS       18
#define WIDGET_SCROLL     19
#define WIDGET_SPLITTER   20
#define WIDGET_MENUBAR    21
#define WIDGET_MENU       22   // a menu: a menubar entry, a submenu, or a popup/context menu
#define WIDGET_MENUITEM   23
#define WIDGET_DIALOG     24
#define WIDGET_IMAGE      25
#define WIDGET_CANVAS     26
#define WIDGET_COLOR      27   // color picker
#define WIDGET_GROUP      28   // titled group box
#define WIDGET_SEPARATOR  29
#define WIDGET_STATUSBAR  30
#define WIDGET_GRID       31
#define WIDGET_STACK      32   // children overlap
#define WIDGET_PANEL      33   // a box with background and border (a frame)
#define WIDGET_TOOLTIP    34   // shown by GuiUi
#define WIDGET_POPUPLIST  35   // a combo box's drop-down list
#define WIDGET_KIND_COUNT 36

// Alignment within the space a container gives a child.
#define WIDGET_ALIGN_START   0
#define WIDGET_ALIGN_CENTER  1
#define WIDGET_ALIGN_END     2
#define WIDGET_ALIGN_STRETCH 3

#define WIDGET_HORIZONTAL 0
#define WIDGET_VERTICAL   1

// Image view fit modes (Widget::set_fit).
#define WIDGET_FIT_CONTAIN 0   // whole image, aspect kept (default)
#define WIDGET_FIT_COVER   1   // fill, aspect kept, cropped
#define WIDGET_FIT_STRETCH 2
#define WIDGET_FIT_NONE    3   // natural size, centered

// Message box buttons (gui_message_box) and dialog results.
#define GUI_MB_OK      1
#define GUI_MB_CANCEL  2
#define GUI_MB_YES     4
#define GUI_MB_NO      8
#define GUI_RESULT_NONE   0
#define GUI_RESULT_OK     1
#define GUI_RESULT_CANCEL 2
#define GUI_RESULT_YES    4
#define GUI_RESULT_NO     8

// ── theme ────────────────────────────────────────────────────────────────

// Colors, fonts and metrics every built-in widget paints with. Metrics are
// logical units (scaled by the window's scale factor). Start from light()
// or dark() and change fields; GuiUi::set_theme() applies one.
struct GuiTheme {
    int dark;
    struct GuiColor window;          // window background
    struct GuiColor panel;           // group boxes, tab pages, menus' background
    struct GuiColor text;
    struct GuiColor textDim;         // placeholders, secondary text
    struct GuiColor textDisabled;
    struct GuiColor accent;          // selection, checked controls, focus ring
    struct GuiColor accentText;      // text on the accent color
    struct GuiColor button;
    struct GuiColor buttonHover;
    struct GuiColor buttonPressed;
    struct GuiColor input;           // text field / list background
    struct GuiColor border;
    struct GuiColor borderStrong;    // hovered / focused borders
    struct GuiColor selection;       // selected list rows, selected text
    struct GuiColor selectionText;
    struct GuiColor hover;           // hovered rows and menu items
    struct GuiColor header;          // table headers, menu bar
    struct GuiColor altRow;          // every other row in tables
    struct GuiColor scrollbar;
    struct GuiColor scrollbarHover;
    struct GuiColor tooltip;
    struct GuiColor tooltipText;
    struct GuiColor shadow;
    struct GuiColor overlay;         // dims the window behind a modal dialog
    float fontSize;
    float smallFontSize;
    float radius;                    // corner radius of controls
    float borderWidth;
    int padding;                     // inside controls
    int spacing;                     // between children of boxes
    int controlHeight;               // buttons, inputs, combo boxes
    int rowHeight;                   // list/tree/table rows, menu items
    int scrollbarSize;
    // light()/dark() pick the system's sans-serif (GuiTypeface::system);
    // set these to GuiTypeface::fallback() for output that doesn't depend
    // on the installed fonts (tests, golden images).
    const ?&GuiTypeface font;        // null: GuiTypeface::fallback()
    const ?&GuiTypeface boldFont;    // null: font

    static struct GuiTheme light();
    static struct GuiTheme dark();
    // The theme matching the system's dark-mode setting where it can be
    // read ($SAFEC_GUI_THEME=dark|light overrides), else light().
    static struct GuiTheme system();
};

// The original per-widget style. Each field, once set through a setter
// (set_bg, set_fg, set_border, set_padding, set_spacing, set_font_scale,
// set_font), overrides the theme for that widget; until then the theme
// decides. 'font' (a bitmap GuiFont) replaces the TrueType theme font when
// set; fontScale then multiplies its cells, and otherwise multiplies the
// theme's font size.
struct WidgetStyle {
    struct GuiColor bg;
    struct GuiColor fg;
    struct GuiColor border;
    int borderWidth;
    int padding;
    int spacing;               // gap between children, for boxes and grids
    int fontScale;             // 1, 2, 3, ...
    const ?&GuiFont font;      // empty (null) = the theme's TrueType font
    int setMask;               // WIDGET_STYLE_* bits: which fields were set
    float fontSize;            // 0 = the theme's
    int bold;
};

#define WIDGET_STYLE_BG        1
#define WIDGET_STYLE_FG        2
#define WIDGET_STYLE_BORDER    4
#define WIDGET_STYLE_PADDING   8
#define WIDGET_STYLE_SPACING   16
#define WIDGET_STYLE_FONT      32

// One node of a tree view.
struct WidgetTreeNode {
    struct String text;
    int parent;          // index of the parent node, -1 for a top-level node
    int expanded;
    int depth;
    long tag;            // yours
};

// One column of a table view.
struct WidgetColumn {
    struct String title;
    int width;           // logical; 0 = share the remaining width
    int align;           // GUI_TEXT_LEFT/CENTER/RIGHT
};

struct Widget {
    int kind;

    // box, computed by layout() each frame (window pixels)
    int x; int y; int w; int h;

    // sizing hints (all optional — 0/unset means "size to content")
    int prefW; int prefH;
    int stretch;    // 0 = natural size; >0 = flex weight along the parent box's main axis
    int alignment;  // WIDGET_ALIGN_* — this widget's position on the parent's cross axis

    struct WidgetStyle style;
    struct String text;
    int visible;
    int enabled;

    int checked;                            // check box, toggle, checkable menu item
    double sliderValue; double sliderMin; double sliderMax; // slider, progress, spinner, splitter (value)
    unsigned long caret;                     // text fields: the caret (a byte offset)

    int hover;
    int pressed;
    int focused;

    // Callback/hook slots: typed functions over an erased context (see
    // std/erased.h). The context's type is each caller's own — 'children'
    // below holds buttons, sliders, labels, ... with unrelated userData
    // types in one tree, which one generic Widget<T> couldn't — so the
    // generic setters (on_click<T> & co.) store fn(?&T) / ?&T as
    // fn(?&Erased) / ?&Erased, and every call passes the pair back together.
    ?fn void(&Widget, ?&Erased) onClick;
    ?&Erased onClickData;
    ?fn void(&Widget, ?&Erased) onChange;
    ?&Erased onChangeData;

    ?fn void(&GuiWindow, &Widget) customDraw;
    ?fn void(&Widget, int, int, int, int) customLayout;
    ?fn int(&Widget, &GuiEvent) customEvent;
    ?&Erased userData;

    struct Vec<&Widget> children;
    ?&Widget parent;        // empty (null) for the root widget

    // ── everything below was added with the full toolkit ──────────────────
    struct String name;          // find() looks widgets up by it
    struct String tooltip;
    struct String placeholder;   // text fields, combo boxes
    struct String shortcut;      // menu items: "Ctrl+S", "Ctrl+Shift+Z", "F5", "Delete"
    int minW; int minH;          // logical; 0 = none
    int maxW; int maxH;
    int marginL; int marginT; int marginR; int marginB;
    int padL; int padT; int padR; int padB;   // -1: use style.padding / the theme
    int valign;                  // stacks and grid cells: vertical alignment (-1: alignment)
    int gridRow; int gridCol; int gridRowSpan; int gridColSpan;
    int columns;                 // grids: number of columns
    struct Vec<int> colStretch;  // grids: per column / per row stretch weights
    struct Vec<int> rowStretch;
    double step;                 // slider/spinner increment (0 = continuous for sliders)
    int decimals;                // spinner/slider value display
    int selected;                // radio, combo, list, tabs, table row, tree node (-1 = none)
    int orientation;             // WIDGET_HORIZONTAL / WIDGET_VERTICAL
    struct Vec<struct String> items;          // radio, combo, list entries; tab titles
    struct Vec<struct WidgetTreeNode> nodes;  // tree view
    struct Vec<struct WidgetColumn> tableCols;
    struct Vec<struct String> cells;          // table: rows x columns, row-major
    int rowCount;
    float scrollX; float scrollY;    // scroll offsets (pixels)
    float contentW; float contentH;  // scrolled content size (pixels)
    unsigned long anchor;        // text fields: the other end of the selection
    float textScroll;            // single-line text: horizontal scroll
    int password;
    int readOnly;
    int wrap;                    // labels, text areas: word wrap
    int indeterminate;           // progress bar: busy animation
    double animPhase;
    int open;                    // combo / menu popup open
    int textAlign;               // GUI_TEXT_LEFT/CENTER/RIGHT
    const ?&GuiImage image;      // image view (the widget doesn't own it)
    int fit;                     // WIDGET_FIT_*
    struct GuiColor color;       // color picker
    float hue; float sat; float val;
    int dragPart;                // what the pointer is dragging (internal)
    float dragX; float dragY;
    double dragValue;
    int hotIndex;                // hovered item / row / tab / menu entry (internal)
    ?fn void(&Widget, ?&Erased) onActivate; ?&Erased onActivateData;
    ?fn void(&Widget, &GuiCanvas, ?&Erased) onDraw; ?&Erased onDrawData;
    // dialogs: run when it closes (internal: message boxes), and a message
    // box's own onClose(dialog, result, userData)
    ?fn void(&Widget, int, ?&Erased) onClose_; ?&Erased onCloseData_;
    ?fn void(&Widget, int, ?&Erased) mbUser_; ?&Erased mbUserData_;
    ?&Widget contextMenu;        // shown on right click
    ?&Widget popup;              // combo: its drop-down list; menu: its open submenu
    ?&Widget owner;              // popups/menus: the widget that opened it
    ?&struct GuiUi uiPtr;        // the GuiUi this tree belongs to (root and overlays)
    int result;                  // dialogs: GUI_RESULT_* when closed
    int focusable;               // -1 = by kind, 0 / 1 = forced
    double clickTime;
    long tag;                    // yours

    // ── constructors ─────────────────────────────────────────────────────
    // All return a freshly heap-allocated, non-null widget — a region-less
    // '&Widget' (see README's "Outliving references"), since nothing here
    // cares whether the tree it ends up in is heap/stack/static/arena-owned.
    // A widget is freed with its tree (free() on the root).
    static &Widget vstack();
    static &Widget hstack();
    static &Widget grid(int columns);
    static &Widget stack();
    static &Widget panel();
    static &Widget group(const &char title);
    static &Widget scroll();
    // Two children (add them with add_child), side by side
    // (WIDGET_HORIZONTAL) or one above the other, with a draggable divider.
    // value() is the first child's share, 0..1.
    static &Widget splitter(int orientation);
    // Pages: add_tab(title, page).
    static &Widget tabs();
    static &Widget spacer();
    static &Widget separator();
    static &Widget label(const &char text);
    static &Widget button(const &char text);
    static &Widget checkbox(const &char text, int checked);
    static &Widget toggle(const &char text, int on);
    // Items with add_item(); selected() / set_selected().
    static &Widget radio_group();
    static &Widget slider(double minValue, double maxValue, double value);
    static &Widget progress(double fraction);
    static &Widget spinner(double minValue, double maxValue, double value, double step);
    static &Widget textinput(const &char placeholder);
    static &Widget textarea();
    static &Widget combo();
    static &Widget list();
    static &Widget tree();
    static &Widget table();
    static &Widget image(const ?&GuiImage img);
    static &Widget color_picker(struct GuiColor initial);
    static &Widget statusbar();
    static &Widget menubar();
    // A menu: inside a menubar, its title is the bar entry; inside another
    // menu, a submenu; on its own, a popup/context menu.
    static &Widget menu(const &char title);
    static &Widget menu_item(const &char text);
    static &Widget menu_separator();
    // A modal dialog box with a title bar; add content and buttons as
    // children (it's a column). Show it with GuiUi::show_modal().
    static &Widget dialog(const &char title);
    // A WIDGET_CANVAS: draw(widget, canvas, userData) paints it; the
    // canvas is already translated to the widget's top-left and clipped to
    // it, in logical units (scaled for the window).
    generic<T>
    static &Widget canvas(fn void(&Widget, &GuiCanvas, ?&T) draw, ?&T userData);
    static &Widget canvas_raw_(fn void(&Widget, &GuiCanvas, ?&Erased) draw, ?&Erased userData);

    // A WIDGET_CUSTOM node drawn and laid out by your own functions (see
    // "Custom widgets" below); userData is the caller's type, stored erased.
    // The parameters are WidgetDrawFn / WidgetLayoutFn / WidgetCustomEventFn
    // (declared after this struct), spelled out: a compiler without forward
    // typedef resolution in struct-body declarations (before SafeC#43)
    // can't match them to the definitions otherwise.
    generic<T>
    static &Widget custom(fn void(&GuiWindow, &Widget) draw, fn void(&Widget, int, int, int, int) layout,
                          fn int(&Widget, &GuiEvent) eventFn, ?&T userData);
    static &Widget custom_raw_(fn void(&GuiWindow, &Widget) draw, fn void(&Widget, int, int, int, int) layout,
                               fn int(&Widget, &GuiEvent) eventFn, ?&Erased userData);

    // ── tree ─────────────────────────────────────────────────────────────
    void add_child(&Widget child);
    // Inserts at 'index' (clamped).
    void insert_child(unsigned long index, &Widget child);
    // Detaches 'child' (not freed: free() it, or add it elsewhere).
    void remove_child(&Widget child);
    // Removes and frees every child.
    void clear_children();
    unsigned long child_count() const;
    ?&Widget child_at(unsigned long i) const;
    // The first widget named 'name' in this subtree (depth first).
    ?&Widget find(const &char name);
    // The GuiUi this widget's tree is attached to (null before the first
    // render/dispatch or GuiUi::new).
    ?&GuiUi ui() const;
    // Marks the widget's window for repainting (and relayout).
    void invalidate();

    // ── customization (each returns this widget, so calls chain) ─────────
    &Widget set_bg(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    &Widget set_fg(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    &Widget set_border(unsigned char r, unsigned char g, unsigned char b,
                       unsigned char a, int width);
    &Widget set_padding(int padding);
    &Widget set_padding4(int left, int top, int right, int bottom);
    &Widget set_margin(int margin);
    &Widget set_margin4(int left, int top, int right, int bottom);
    &Widget set_spacing(int spacing);
    &Widget set_pref_size(int prefW, int prefH);
    &Widget set_min_size(int minW, int minH);
    &Widget set_max_size(int maxW, int maxH);
    &Widget set_stretch(int stretch);
    &Widget set_alignment(int alignment);
    // Both axes for stacks and grid cells (horizontal, vertical).
    &Widget set_align2(int halign, int valign);
    &Widget set_font_scale(int scale);
    &Widget set_font(const ?&GuiFont font);
    // TrueType font size (logical px) and weight for this widget's text.
    &Widget set_font_size(float size);
    &Widget set_bold(int bold);
    &Widget set_visible(int visible);
    &Widget set_enabled(int enabled);
    &Widget set_text(const &char text);
    &Widget set_name(const &char name);
    &Widget set_tooltip(const &char text);
    &Widget set_placeholder(const &char text);
    &Widget set_text_align(int align);
    &Widget set_wrap(int wrap);
    &Widget set_password(int on);
    &Widget set_read_only(int on);
    &Widget set_focusable(int on);
    // Grid placement (and spans of 1 or more cells).
    &Widget set_cell(int row, int col);
    &Widget set_span(int rows, int cols);
    &Widget set_column_stretch(int col, int stretch);
    &Widget set_row_stretch(int row, int stretch);
    &Widget set_orientation(int orientation);
    &Widget set_context_menu(?&Widget menu);
    &Widget set_tag(long tag);

    // ── values ───────────────────────────────────────────────────────────
    // slider / progress / spinner / splitter; clamped to the range, rounded
    // to 'step' when it's > 0. Setting a value doesn't fire on_change.
    &Widget set_value(double v);
    double  value() const;
    &Widget set_range(double minValue, double maxValue);
    &Widget set_step(double step);
    &Widget set_decimals(int decimals);
    &Widget set_indeterminate(int on);
    // check box / toggle / menu item
    &Widget set_checked(int checked);
    int     is_checked() const;
    // items of radio groups, combo boxes and list views
    &Widget add_item(const &char text);
    void    clear_items();
    unsigned long item_count() const;
    const &char item_text(unsigned long i) const;   // "" out of range
    &Widget set_selected(int index);
    int     selected_index() const;
    // tabs
    &Widget add_tab(const &char title, &Widget page);
    // tree views: a node under 'parent' (-1: top level); returns its index
    int     add_node(int parent, const &char text);
    &Widget set_expanded(int node, int expanded);
    int     node_count() const;
    const &char node_text(int node) const;
    // table views
    &Widget add_column(const &char title, int width);
    int     add_row();                               // returns the row index
    &Widget set_cell_text(int row, int col, const &char text);
    const &char cell_text(int row, int col) const;
    int     row_count() const;
    void    clear_rows();
    // text fields: the selection (byte offsets, start <= end)
    &Widget select(unsigned long start, unsigned long end);
    &Widget select_all();
    unsigned long selection_start() const;
    unsigned long selection_end() const;
    // image views
    &Widget set_image(const ?&GuiImage img);
    &Widget set_fit(int fit);
    // color pickers
    &Widget set_color(struct GuiColor c);
    struct GuiColor get_color() const;
    // menu items: the shortcut shown and triggered ("Ctrl+O", "Ctrl+Shift+S",
    // "F2", "Delete"; Ctrl means Command on macOS)
    &Widget set_shortcut(const &char shortcut);
    // dialogs: closes it (GuiUi::close_modal) with 'result'
    void    close_dialog(int result);

    // ── callbacks (each returns this widget, so calls chain) ─────────────
    // cb(widget, userData). userData's type is the caller's; the widget
    // stores it erased (see the field comment above).
    generic<T> &Widget on_click(fn void(&Widget, ?&T) cb, ?&T userData);
    generic<T> &Widget on_change(fn void(&Widget, ?&T) cb, ?&T userData);
    generic<T> &Widget on_activate(fn void(&Widget, ?&T) cb, ?&T userData);
    generic<T> &Widget on_draw(fn void(&Widget, &GuiCanvas, ?&T) cb, ?&T userData);
    &Widget on_click_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData);
    &Widget on_change_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData);
    &Widget on_activate_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData);
    &Widget on_draw_raw_(fn void(&Widget, &GuiCanvas, ?&Erased) cb, ?&Erased userData);

    // ── per-frame pipeline (for driving a tree by hand) ──────────────────

    // Positions this widget and its whole subtree within the box
    // (x,y,w,h) — window pixels.
    void layout(int x, int y, int w, int h);

    // Lays the tree out to the window's size and paints it into 'win'.
    // (A tree that's already laid out to another box — layout() called
    // with a box not at the window's origin — keeps that box.)
    void render(&GuiWindow win);

    // Routes one event through the tree rooted here: hover/pressed/focus
    // state, clicks, value changes, text editing, scrolling, popups, and
    // a WIDGET_CUSTOM node's customEvent hook. Returns 1 if some widget
    // consumed the event.
    //
    // After that, for GUI_EVENT_KEY_DOWN, a default action runs — the same
    // idea as a browser's default DOM action for a key event — *unless*
    // ev.prevent_default() (see gui.h) was called on this event first
    // (typically from a WIDGET_CUSTOM node's WidgetCustomEventFn, the one
    // hook that gets the raw GuiEvent* and can act before the default
    // fires):
    //   Tab / Shift+Tab   moves focus to the next/previous focusable widget (wraps around)
    //   Enter / Space     activates the focused widget (button click / check box toggle)
    //   Escape            closes a popup or dialog, or clears focus
    //   Left/Right        nudges a focused slider by 5% of its range (or its step)
    // These are genuinely *default* actions: they still run even if
    // dispatch already returned 1 for this event (e.g. a focused text
    // input consuming Backspace doesn't block Tab from later moving focus
    // off it).
    int dispatch_event(&GuiEvent ev);

    // Recursively frees this widget and its whole subtree (children,
    // text, style, context menu), including the widget's own allocation.
    // Detach it from its parent first (remove_child) when it has one.
    void free();
};

// onClick/onChange/onActivate(widget, userData).
typedef fn void(&Widget, ?&Erased) WidgetCallback;
// A canvas widget's draw(widget, canvas, userData).
typedef fn void(&Widget, &GuiCanvas, ?&Erased) WidgetCanvasFn;
// Custom draw(window, widget) — called instead of the built-in renderer
// for a WIDGET_CUSTOM node; must draw within [widget->x, widget->x+widget->w) x
// [widget->y, widget->y+widget->h) using GuiWindow's drawing methods (win.fill_rect(), win.draw_text(), ...).
typedef fn void(&GuiWindow, &Widget) WidgetDrawFn;
// Custom layout(widget, x, y, w, h) — called instead of the built-in
// layout for a WIDGET_CUSTOM node; must set widget->x/y/w/h (typically to
// the given box) and lay out+recurse into any children itself.
typedef fn void(&Widget, int, int, int, int) WidgetLayoutFn;
// Custom event(widget, event) -> 1 if handled (stops the event from
// reaching this widget's parent/siblings), 0 to let normal dispatch
// (hover/press/click detection) continue.
typedef fn int(&Widget, &GuiEvent) WidgetCustomEventFn;

// ── the UI context ───────────────────────────────────────────────────────

// Everything a widget tree needs that isn't a widget: the window it's
// shown in, the theme and scale, keyboard focus, the pointer, open popups
// (menus, drop-down lists, tooltips) and modal dialogs, and the canvas it
// paints with. One per window. GuiApp makes them (GuiApp::set_root);
// Widget::render/dispatch_event make one on first use for a root that has
// none; or make one yourself.
struct GuiUi {
    &Widget root;
    ?&GuiWindow win;              // set by attach() / render()
    struct GuiTheme theme;
    float scale;                  // the window's scale factor
    struct GuiCanvas canvas;
    int canvasReady;
    struct Vec<&Widget> overlays; // popups and dialogs, bottom to top
    ?&Widget focus;
    ?&Widget hoverW;
    ?&Widget capture;             // gets the pointer while a button is held
    float mouseX; float mouseY;
    int mouseInside;
    ?&Widget tipFor;              // tooltip candidate and since when
    double tipSince;
    &Widget tip;                  // the tooltip widget
    int tipShown;
    double nowMs;
    int needsLayout;
    int dirty;
    int layoutW; int layoutH;     // the size the tree was last laid out to
    int cursor;

    // A context for 'root' (attached to it: root.ui() returns it).
    static &GuiUi new(&Widget root);
    // The window it draws into and gets events from.
    void attach(&GuiWindow win);
    void set_theme(struct GuiTheme t);
    // Routes one event (see Widget::dispatch_event). Returns 1 if consumed.
    int  dispatch(&GuiEvent ev);
    // Lays out (if needed) and paints the tree and its overlays into the
    // attached window's buffer.
    void render();
    // Advances time-driven state — tooltips, caret blinking, busy
    // indicators — to 'nowMs' (gui_time_ms()). Marks the UI dirty when
    // something visible changed.
    void tick(double nowMs);
    // 1 while something animates (the next frame will look different).
    int  animating() const;
    // 1 when the UI needs repainting; render() clears it.
    int  needs_redraw() const;
    void invalidate();
    void request_layout();
    void set_focus(?&Widget w);
    // Shows a menu (or any widget) as a popup at window position (x, y),
    // kept on screen. A click outside closes it.
    void popup(&Widget menu, float x, float y);
    void close_popups();
    // Shows 'dialog' centered over the window, blocking input to
    // everything under it until close_modal() (or the dialog's
    // close_dialog()). The UI doesn't own a shown dialog: free it after
    // closing it (or reuse it).
    void show_modal(&Widget dialog);
    void close_modal(&Widget dialog);
    // The topmost open dialog, or null.
    ?&Widget modal() const;
    // Frees the UI and its whole widget tree.
    void free();
};

// A standard message box: 'text' and a row of buttons (GUI_MB_* bits),
// shown modally. onClose(dialog, result) runs once a button (result
// GUI_RESULT_*) or Escape (GUI_RESULT_CANCEL) closed it; the box frees
// itself afterwards. The buttons are named "yes", "no", "ok" and "cancel"
// (Widget::find).
generic<T>
void gui_message_box(&GuiUi ui, const &char title, const &char text, int buttons,
                     fn void(&Widget, int, ?&T) onClose, ?&T userData);
void gui_message_box_raw_(&GuiUi ui, const &char title, const &char text, int buttons,
                          ?fn void(&Widget, int, ?&Erased) onClose, ?&Erased userData);

// Custom widgets: 'draw'/'layout' are required; 'eventFn' may be NULL (the
// widget then just participates in hit-testing/hover/press/click like a
// button would, with onClick/onChange still firing normally — set eventFn
// only when you need to intercept raw events yourself, e.g. for a custom
// slider/canvas that tracks drag deltas).
generic<T>
&Widget Widget::custom(WidgetDrawFn draw, WidgetLayoutFn layout,
                       WidgetCustomEventFn eventFn, ?&T userData) {
    unsafe { return Widget::custom_raw_(draw, layout, eventFn, (?&Erased)userData); }
}

generic<T>
&Widget Widget::canvas(fn void(&Widget, &GuiCanvas, ?&T) draw, ?&T userData) {
    unsafe { return Widget::canvas_raw_((fn void(&Widget, &GuiCanvas, ?&Erased))draw, (?&Erased)userData); }
}

generic<T>
&Widget Widget::on_click(fn void(&Widget, ?&T) cb, ?&T userData) {
    unsafe { return self.on_click_raw_((fn void(&Widget, ?&Erased))cb, (?&Erased)userData); }
}

generic<T>
&Widget Widget::on_change(fn void(&Widget, ?&T) cb, ?&T userData) {
    unsafe { return self.on_change_raw_((fn void(&Widget, ?&Erased))cb, (?&Erased)userData); }
}

generic<T>
&Widget Widget::on_activate(fn void(&Widget, ?&T) cb, ?&T userData) {
    unsafe { return self.on_activate_raw_((fn void(&Widget, ?&Erased))cb, (?&Erased)userData); }
}

generic<T>
&Widget Widget::on_draw(fn void(&Widget, &GuiCanvas, ?&T) cb, ?&T userData) {
    unsafe { return self.on_draw_raw_((fn void(&Widget, &GuiCanvas, ?&Erased))cb, (?&Erased)userData); }
}

generic<T>
void gui_message_box(&GuiUi ui, const &char title, const &char text, int buttons,
                     fn void(&Widget, int, ?&T) onClose, ?&T userData) {
    unsafe {
        gui_message_box_raw_(ui, title, text, buttons, (?fn void(&Widget, int, ?&Erased))onClose, (?&Erased)userData);
    }
}

} // namespace std
