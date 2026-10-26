// SafeC Standard Library — GUI: the widget toolkit's core (see
// gui_widget.h): widget construction and properties, themes, measuring and
// layout dispatch, painting, hit testing, event routing, focus, popups,
// dialogs, tooltips and GuiUi. The widget families live in
// gui_widget_basic.sc, gui_widget_text.sc, gui_widget_list.sc,
// gui_widget_layout.sc and gui_widget_menu.sc.
#pragma once
#include <std/stddef.h>
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/gui/gui_font.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_widget_internal.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
extern void* memset(void* ptr, int val, size_t n);
extern char* getenv(const char* name);

namespace std {

// ── themes ────────────────────────────────────────────────────────────────

static void __gui_theme_metrics(&GuiTheme t) {
    t.fontSize = 14.0f;
    t.smallFontSize = 12.0f;
    t.radius = 4.0f;
    t.borderWidth = 1.0f;
    t.padding = 6;
    t.spacing = 8;
    t.controlHeight = 28;
    t.rowHeight = 24;
    t.scrollbarSize = 10;
    t.font = GuiTypeface::system("sans");      // cached after the first lookup
    t.boldFont = GuiTypeface::system_bold("sans");
}

struct GuiTheme GuiTheme::light() {
    struct GuiTheme t;
    t.dark = 0;
    t.window = GuiColor::hex(0xF3F3F3U);
    t.panel = GuiColor::hex(0xFBFBFBU);
    t.text = GuiColor::hex(0x1E1E1EU);
    t.textDim = GuiColor::hex(0x707070U);
    t.textDisabled = GuiColor::hex(0xA6A6A6U);
    t.accent = GuiColor::hex(0x2F6FEBU);
    t.accentText = GuiColor::hex(0xFFFFFFU);
    t.button = GuiColor::hex(0xFFFFFFU);
    t.buttonHover = GuiColor::hex(0xEEF2FAU);
    t.buttonPressed = GuiColor::hex(0xDCE4F5U);
    t.input = GuiColor::hex(0xFFFFFFU);
    t.border = GuiColor::hex(0xC4C4C4U);
    t.borderStrong = GuiColor::hex(0x8A8A8AU);
    t.selection = GuiColor::hex(0x2F6FEBU);
    t.selectionText = GuiColor::hex(0xFFFFFFU);
    t.hover = GuiColor::hex(0xE6ECF8U);
    t.header = GuiColor::hex(0xEAEAEAU);
    t.altRow = GuiColor::hex(0xF6F6F6U);
    t.scrollbar = GuiColor::hexa(0x00000040U);
    t.scrollbarHover = GuiColor::hexa(0x00000078U);
    t.tooltip = GuiColor::hex(0x2B2B2BU);
    t.tooltipText = GuiColor::hex(0xFFFFFFU);
    t.shadow = GuiColor::hexa(0x00000030U);
    t.overlay = GuiColor::hexa(0x00000050U);
    __gui_theme_metrics(&t);
    return t;
}

struct GuiTheme GuiTheme::dark() {
    struct GuiTheme t;
    t.dark = 1;
    t.window = GuiColor::hex(0x1F1F1FU);
    t.panel = GuiColor::hex(0x282828U);
    t.text = GuiColor::hex(0xE6E6E6U);
    t.textDim = GuiColor::hex(0x9A9A9AU);
    t.textDisabled = GuiColor::hex(0x5E5E5EU);
    t.accent = GuiColor::hex(0x4C8DFFU);
    t.accentText = GuiColor::hex(0xFFFFFFU);
    t.button = GuiColor::hex(0x333333U);
    t.buttonHover = GuiColor::hex(0x3D3D3DU);
    t.buttonPressed = GuiColor::hex(0x2A2A2AU);
    t.input = GuiColor::hex(0x262626U);
    t.border = GuiColor::hex(0x4A4A4AU);
    t.borderStrong = GuiColor::hex(0x7A7A7AU);
    t.selection = GuiColor::hex(0x3A6FD8U);
    t.selectionText = GuiColor::hex(0xFFFFFFU);
    t.hover = GuiColor::hex(0x353A45U);
    t.header = GuiColor::hex(0x2D2D2DU);
    t.altRow = GuiColor::hex(0x252525U);
    t.scrollbar = GuiColor::hexa(0xFFFFFF40U);
    t.scrollbarHover = GuiColor::hexa(0xFFFFFF78U);
    t.tooltip = GuiColor::hex(0xF0F0F0U);
    t.tooltipText = GuiColor::hex(0x202020U);
    t.shadow = GuiColor::hexa(0x00000080U);
    t.overlay = GuiColor::hexa(0x00000080U);
    __gui_theme_metrics(&t);
    return t;
}

struct GuiTheme GuiTheme::system() {
    int dark = 0;
    unsafe {
        char* env = getenv("SAFEC_GUI_THEME");
        if (env != (char*)0 && env[0] == 'd') { dark = 1; }
        if (env == (char*)0) {
            char* gtk = getenv("GTK_THEME");
            if (gtk != (char*)0) {
                unsigned long i = 0UL;
                while (gtk[i] != '\0') {
                    if (gtk[i] == ':' && gtk[i + 1UL] == 'd') { dark = 1; }
                    i = i + 1UL;
                }
            }
        }
    }
    return dark ? GuiTheme::dark() : GuiTheme::light();
}

// ── allocation and construction ───────────────────────────────────────────

static int __w_default_align(int kind) {
    if (kind == WIDGET_TEXTINPUT || kind == WIDGET_TEXTAREA || kind == WIDGET_SLIDER ||
        kind == WIDGET_PROGRESS || kind == WIDGET_LIST || kind == WIDGET_TREE || kind == WIDGET_TABLE ||
        kind == WIDGET_TABS || kind == WIDGET_SCROLL || kind == WIDGET_SPLITTER || kind == WIDGET_SEPARATOR ||
        kind == WIDGET_MENUBAR || kind == WIDGET_STATUSBAR || kind == WIDGET_GROUP || kind == WIDGET_PANEL ||
        kind == WIDGET_GRID || kind == WIDGET_CANVAS || kind == WIDGET_STACK || kind == WIDGET_VSTACK || kind == WIDGET_HSTACK) {
        return WIDGET_ALIGN_STRETCH;
    }
    return WIDGET_ALIGN_START;
}

static &Widget __widget_new(int kind) {
    &Widget w;
    unsafe {
        w = (struct Widget*)alloc_zeroed_raw_(sizeof(struct Widget));
    }
    w.kind = kind;
    w.x = 0; w.y = 0; w.w = 0; w.h = 0;
    w.prefW = 0; w.prefH = 0;
    w.stretch = 0;
    w.alignment = __w_default_align(kind);
    w.style.bg = GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0);
    w.style.fg = GuiColor::rgba((unsigned char)230, (unsigned char)230, (unsigned char)235, (unsigned char)255);
    w.style.border = GuiColor::rgba((unsigned char)110, (unsigned char)110, (unsigned char)120, (unsigned char)255);
    w.style.borderWidth = 0;
    w.style.padding = 0;
    w.style.spacing = 0;
    w.style.fontScale = 1;
    w.style.font = null;
    w.style.setMask = 0;
    w.style.fontSize = 0.0f;
    w.style.bold = 0;
    w.text = String::new();
    w.visible = 1;
    w.enabled = 1;
    w.checked = 0;
    w.sliderValue = 0.0;
    w.sliderMin = 0.0;
    w.sliderMax = 1.0;
    w.caret = 0UL;
    w.hover = 0; w.pressed = 0; w.focused = 0;
    unsafe {
        w.onClick = null; w.onClickData = null;
        w.onChange = null; w.onChangeData = null;
        w.customDraw = null; w.customLayout = null; w.customEvent = null; w.userData = null;
        w.onActivate = null; w.onActivateData = null;
        w.onDraw = null; w.onDrawData = null;
        w.onClose_ = null; w.onCloseData_ = null; w.mbUser_ = null; w.mbUserData_ = null;
        w.uiPtr = null;
    }
    w.children = Vec::new();
    w.parent = null;
    w.name = String::new();
    w.tooltip = String::new();
    w.placeholder = String::new();
    w.shortcut = String::new();
    w.minW = 0; w.minH = 0; w.maxW = 0; w.maxH = 0;
    w.marginL = 0; w.marginT = 0; w.marginR = 0; w.marginB = 0;
    w.padL = -1; w.padT = -1; w.padR = -1; w.padB = -1;
    w.valign = -1;
    w.gridRow = -1; w.gridCol = -1; w.gridRowSpan = 1; w.gridColSpan = 1;
    w.columns = 1;
    w.colStretch = Vec::new();
    w.rowStretch = Vec::new();
    w.step = 0.0;
    w.decimals = 0;
    w.selected = -1;
    w.orientation = WIDGET_HORIZONTAL;
    w.items = Vec::new();
    w.nodes = Vec::new();
    w.tableCols = Vec::new();
    w.cells = Vec::new();
    w.rowCount = 0;
    w.scrollX = 0.0f; w.scrollY = 0.0f; w.contentW = 0.0f; w.contentH = 0.0f;
    w.anchor = 0UL;
    w.textScroll = 0.0f;
    w.password = 0; w.readOnly = 0; w.wrap = 0;
    w.indeterminate = 0;
    w.animPhase = 0.0;
    w.open = 0;
    w.textAlign = GUI_TEXT_LEFT;
    w.image = null;
    w.fit = WIDGET_FIT_CONTAIN;
    w.color = GuiColor::hex(0x2F6FEBU);
    w.hue = 0.0f; w.sat = 0.0f; w.val = 0.0f;
    w.dragPart = 0; w.dragX = 0.0f; w.dragY = 0.0f; w.dragValue = 0.0;
    w.hotIndex = -1;
    w.contextMenu = null;
    w.popup = null;
    w.owner = null;
    w.result = GUI_RESULT_NONE;
    w.focusable = -1;
    w.clickTime = -1000000.0;
    w.tag = 0L;
    return w;
}

// 'self' as a plain (region-less) reference: methods hand it out in their
// return values and store it in other widgets' fields.
static &Widget __w_self(&Widget w) {
    unsafe { return (struct Widget*)w; }
}

static &Widget __widget_new_text(int kind, const &char text) {
    &Widget w = __widget_new(kind);
    w.text.free();
    w.text = String::from(text);
    return w;
}

&Widget Widget::vstack() {
    &Widget w = __widget_new(WIDGET_VSTACK);
    w.orientation = WIDGET_VERTICAL;
    return w;
}
&Widget Widget::hstack() { return __widget_new(WIDGET_HSTACK); }
&Widget Widget::grid(int columns) {
    &Widget w = __widget_new(WIDGET_GRID);
    w.columns = columns < 1 ? 1 : columns;
    return w;
}
&Widget Widget::stack() { return __widget_new(WIDGET_STACK); }
&Widget Widget::panel() {
    &Widget w = __widget_new(WIDGET_PANEL);
    w.orientation = WIDGET_VERTICAL;
    return w;
}
&Widget Widget::group(const &char title) {
    &Widget w = __widget_new_text(WIDGET_GROUP, title);
    w.orientation = WIDGET_VERTICAL;
    return w;
}
&Widget Widget::scroll() { return __widget_new(WIDGET_SCROLL); }
&Widget Widget::splitter(int orientation) {
    &Widget w = __widget_new(WIDGET_SPLITTER);
    w.orientation = orientation;
    w.sliderValue = 0.5;
    w.sliderMin = 0.0;
    w.sliderMax = 1.0;
    return w;
}
&Widget Widget::tabs() {
    &Widget w = __widget_new(WIDGET_TABS);
    w.selected = 0;
    return w;
}
&Widget Widget::spacer() {
    &Widget w = __widget_new(WIDGET_SPACER);
    w.stretch = 1;
    return w;
}
&Widget Widget::separator() { return __widget_new(WIDGET_SEPARATOR); }
&Widget Widget::label(const &char text) { return __widget_new_text(WIDGET_LABEL, text); }
&Widget Widget::button(const &char text) { return __widget_new_text(WIDGET_BUTTON, text); }
&Widget Widget::checkbox(const &char text, int checked) {
    &Widget w = __widget_new_text(WIDGET_CHECKBOX, text);
    w.checked = checked ? 1 : 0;
    return w;
}
&Widget Widget::toggle(const &char text, int on) {
    &Widget w = __widget_new_text(WIDGET_TOGGLE, text);
    w.checked = on ? 1 : 0;
    return w;
}
&Widget Widget::radio_group() {
    &Widget w = __widget_new(WIDGET_RADIO);
    w.orientation = WIDGET_VERTICAL;
    w.selected = -1;
    return w;
}
&Widget Widget::slider(double minValue, double maxValue, double value) {
    &Widget w = __widget_new(WIDGET_SLIDER);
    w.sliderMin = minValue;
    w.sliderMax = maxValue;
    w.sliderValue = value;
    return w;
}
&Widget Widget::progress(double fraction) {
    &Widget w = __widget_new(WIDGET_PROGRESS);
    w.sliderMin = 0.0;
    w.sliderMax = 1.0;
    w.sliderValue = __w_clamp_value(w, fraction);
    return w;
}
&Widget Widget::spinner(double minValue, double maxValue, double value, double step) {
    &Widget w = __widget_new(WIDGET_SPINNER);
    w.sliderMin = minValue;
    w.sliderMax = maxValue;
    w.step = step > 0.0 ? step : 1.0;
    w.sliderValue = __w_clamp_value(w, value);
    __w_format_value(w, w.sliderValue, &w.text);
    return w;
}
&Widget Widget::textinput(const &char placeholder) {
    &Widget w = __widget_new(WIDGET_TEXTINPUT);
    w.placeholder.free();
    w.placeholder = String::from(placeholder);
    return w;
}
&Widget Widget::textarea() {
    &Widget w = __widget_new(WIDGET_TEXTAREA);
    w.wrap = 1;
    return w;
}
&Widget Widget::combo() { return __widget_new(WIDGET_COMBO); }
&Widget Widget::list() { return __widget_new(WIDGET_LIST); }
&Widget Widget::tree() { return __widget_new(WIDGET_TREE); }
&Widget Widget::table() { return __widget_new(WIDGET_TABLE); }
&Widget Widget::image(const ?&GuiImage img) {
    &Widget w = __widget_new(WIDGET_IMAGE);
    w.image = img;
    return w;
}
&Widget Widget::color_picker(struct GuiColor initial) {
    &Widget w = __widget_new(WIDGET_COLOR);
    w.set_color(initial);
    return w;
}
&Widget Widget::statusbar() { return __widget_new(WIDGET_STATUSBAR); }
&Widget Widget::menubar() { return __widget_new(WIDGET_MENUBAR); }
&Widget Widget::menu(const &char title) {
    &Widget w = __widget_new_text(WIDGET_MENU, title);
    w.orientation = WIDGET_VERTICAL;
    return w;
}
&Widget Widget::menu_item(const &char text) { return __widget_new_text(WIDGET_MENUITEM, text); }
&Widget Widget::menu_separator() {
    &Widget w = __widget_new(WIDGET_MENUITEM);
    w.orientation = WIDGET_VERTICAL;   // marks a separator item
    w.enabled = 0;
    return w;
}
&Widget Widget::dialog(const &char title) {
    &Widget w = __widget_new_text(WIDGET_DIALOG, title);
    w.orientation = WIDGET_VERTICAL;
    return w;
}
&Widget Widget::canvas_raw_(fn void(&Widget, &GuiCanvas, ?&Erased) draw, ?&Erased userData) {
    &Widget w = __widget_new(WIDGET_CANVAS);
    unsafe {
        w.onDraw = draw;
        w.onDrawData = userData;
    }
    return w;
}
&Widget Widget::custom_raw_(WidgetDrawFn draw, WidgetLayoutFn layout,
                            WidgetCustomEventFn eventFn, ?&Erased userData) {
    &Widget w = __widget_new(WIDGET_CUSTOM);
    unsafe {
        w.customDraw = draw;
        w.customLayout = layout;
        w.customEvent = eventFn;
        w.userData = userData;
    }
    return w;
}

// ── the tree ──────────────────────────────────────────────────────────────

static void __w_mark(&Widget w) {
    ?&GuiUi u = w.ui();
    match (u) {
        case some(ui): { ui.needsLayout = 1; ui.dirty = 1; }
        case null: { }
    }
}

void Widget::add_child(&Widget child) {
    child.parent = __w_self(self);
    self.children.push(child);
    __w_mark(self);
}

void Widget::insert_child(unsigned long index, &Widget child) {
    child.parent = __w_self(self);
    if (index > self.children.len) { index = self.children.len; }
    self.children.insert(index, child);
    __w_mark(self);
}

static int __w_same(const ?&Widget a, const &Widget b) {
    unsafe { return (const void*)a == (const void*)b ? 1 : 0; }
}

// Forgets references to 'gone' (and its subtree) held by the UI.
static void __w_ui_forget(&Widget gone);

void Widget::remove_child(&Widget child) {
    unsigned long i = 0UL;
    while (i < self.children.len) {
        if (__w_same(*self.children.at(i), child)) {
            __w_ui_forget(child);
            self.children.remove(i);
            child.parent = null;
            __w_mark(self);
            return;
        }
        i = i + 1UL;
    }
}

void Widget::clear_children() {
    while (self.children.len > 0UL) {
        &Widget c = *self.children.at(self.children.len - 1UL);
        self.remove_child(c);
        c.free();
    }
}

unsigned long Widget::child_count() const { return self.children.len; }

?&Widget Widget::child_at(unsigned long i) const {
    if (i >= self.children.len) { return null; }
    return *self.children.at(i);
}

?&Widget Widget::find(const &char name) {
    if (self.name.eq_cstr(name)) { return __w_self(self); }
    unsigned long i = 0UL;
    while (i < self.children.len) {
        &Widget c = *self.children.at(i);
        ?&Widget f = c.find(name);
        if (!f.is_null()) { return f; }
        i = i + 1UL;
    }
    return null;
}

?&GuiUi Widget::ui() const {
    const ?&Widget w = self;
    int guard = 0;
    while (guard < 10000) {
        match (w) {
            case null: return null;
            case some(cur): {
                unsafe {
                    if (!cur.uiPtr.is_null()) { return cur.uiPtr; }
                }
                // popups belong to their owner's UI
                if (cur.parent.is_null() && !cur.owner.is_null()) { w = cur.owner; }
                else { w = cur.parent; }
            }
        }
        guard = guard + 1;
    }
    return null;
}

void Widget::invalidate() { __w_mark(self); }

// ── setters ───────────────────────────────────────────────────────────────

&Widget Widget::set_bg(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    self.style.bg = GuiColor::rgba(r, g, b, a);
    self.style.setMask = self.style.setMask | WIDGET_STYLE_BG;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_fg(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    self.style.fg = GuiColor::rgba(r, g, b, a);
    self.style.setMask = self.style.setMask | WIDGET_STYLE_FG;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_border(unsigned char r, unsigned char g, unsigned char b, unsigned char a, int width) {
    self.style.border = GuiColor::rgba(r, g, b, a);
    self.style.borderWidth = width;
    self.style.setMask = self.style.setMask | WIDGET_STYLE_BORDER;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_padding(int padding) {
    self.style.padding = padding;
    self.style.setMask = self.style.setMask | WIDGET_STYLE_PADDING;
    self.padL = -1; self.padT = -1; self.padR = -1; self.padB = -1;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_padding4(int left, int top, int right, int bottom) {
    self.padL = left; self.padT = top; self.padR = right; self.padB = bottom;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_margin(int margin) { return self.set_margin4(margin, margin, margin, margin); }
&Widget Widget::set_margin4(int left, int top, int right, int bottom) {
    self.marginL = left; self.marginT = top; self.marginR = right; self.marginB = bottom;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_spacing(int spacing) {
    self.style.spacing = spacing;
    self.style.setMask = self.style.setMask | WIDGET_STYLE_SPACING;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_pref_size(int prefW, int prefH) { self.prefW = prefW; self.prefH = prefH; __w_mark(self); return __w_self(self); }
&Widget Widget::set_min_size(int minW, int minH) { self.minW = minW; self.minH = minH; __w_mark(self); return __w_self(self); }
&Widget Widget::set_max_size(int maxW, int maxH) { self.maxW = maxW; self.maxH = maxH; __w_mark(self); return __w_self(self); }
&Widget Widget::set_stretch(int stretch) { self.stretch = stretch; __w_mark(self); return __w_self(self); }
&Widget Widget::set_alignment(int alignment) { self.alignment = alignment; __w_mark(self); return __w_self(self); }
&Widget Widget::set_align2(int halign, int valign) {
    self.alignment = halign;
    self.valign = valign;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_font_scale(int scale) {
    self.style.fontScale = scale < 1 ? 1 : scale;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_font(const ?&GuiFont font) {
    self.style.font = font;
    self.style.setMask = self.style.setMask | WIDGET_STYLE_FONT;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_font_size(float size) { self.style.fontSize = size; __w_mark(self); return __w_self(self); }
&Widget Widget::set_bold(int bold) { self.style.bold = bold; __w_mark(self); return __w_self(self); }
&Widget Widget::set_visible(int visible) {
    self.visible = visible ? 1 : 0;
    if (!self.visible) { __w_ui_forget(self); }
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_enabled(int enabled) {
    self.enabled = enabled ? 1 : 0;
    if (!self.enabled) { self.pressed = 0; }
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_text(const &char text) {
    self.text.free();
    self.text = String::from(text);
    if (self.caret > self.text.len) { self.caret = self.text.len; }
    if (self.anchor > self.text.len) { self.anchor = self.text.len; }
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_name(const &char name) { self.name.free(); self.name = String::from(name); return __w_self(self); }
&Widget Widget::set_tooltip(const &char text) { self.tooltip.free(); self.tooltip = String::from(text); return __w_self(self); }
&Widget Widget::set_placeholder(const &char text) {
    self.placeholder.free();
    self.placeholder = String::from(text);
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::set_text_align(int align) { self.textAlign = align; __w_mark(self); return __w_self(self); }
&Widget Widget::set_wrap(int wrap) { self.wrap = wrap; __w_mark(self); return __w_self(self); }
&Widget Widget::set_password(int on) { self.password = on; __w_mark(self); return __w_self(self); }
&Widget Widget::set_read_only(int on) { self.readOnly = on; __w_mark(self); return __w_self(self); }
&Widget Widget::set_focusable(int on) { self.focusable = on ? 1 : 0; return __w_self(self); }
&Widget Widget::set_cell(int row, int col) { self.gridRow = row; self.gridCol = col; __w_mark(self); return __w_self(self); }
&Widget Widget::set_span(int rows, int cols) {
    self.gridRowSpan = rows < 1 ? 1 : rows;
    self.gridColSpan = cols < 1 ? 1 : cols;
    __w_mark(self);
    return __w_self(self);
}
static void __w_set_weight(&Vec<int> v, int index, int stretch) {
    if (index < 0) { return; }
    while (v.len <= (unsigned long)index) { v.push(0); }
    v.set((unsigned long)index, stretch);
}
&Widget Widget::set_column_stretch(int col, int stretch) { __w_set_weight(&self.colStretch, col, stretch); __w_mark(self); return __w_self(self); }
&Widget Widget::set_row_stretch(int row, int stretch) { __w_set_weight(&self.rowStretch, row, stretch); __w_mark(self); return __w_self(self); }
&Widget Widget::set_orientation(int orientation) { self.orientation = orientation; __w_mark(self); return __w_self(self); }
&Widget Widget::set_context_menu(?&Widget menu) {
    match (self.contextMenu) {
        case some(old): { old.owner = null; }
        case null: { }
    }
    self.contextMenu = menu;
    match (menu) {
        case some(m): { m.owner = __w_self(self); }
        case null: { }
    }
    return __w_self(self);
}
&Widget Widget::set_tag(long tag) { self.tag = tag; return __w_self(self); }

// ── values ────────────────────────────────────────────────────────────────

double __w_clamp_value(const &Widget w, double v) {
    double lo = w.sliderMin < w.sliderMax ? w.sliderMin : w.sliderMax;
    double hi = w.sliderMin < w.sliderMax ? w.sliderMax : w.sliderMin;
    if (w.step > 0.0) {
        double k = floor_d((v - w.sliderMin) / w.step + 0.5);
        v = w.sliderMin + k * w.step;
    }
    if (v < lo) { v = lo; }
    if (v > hi) { v = hi; }
    return v;
}

void __w_format_value(const &Widget w, double v, &String out) {
    out.clear();
    if (w.decimals <= 0) {
        long long iv = (long long)(v < 0.0 ? v - 0.5 : v + 0.5);
        out.push_int(iv);
    } else {
        out.push_float(v, w.decimals);
    }
}

&Widget Widget::set_value(double v) {
    self.sliderValue = __w_clamp_value(self, v);
    if (self.kind == WIDGET_SPINNER) { __w_format_value(self, self.sliderValue, &self.text); self.caret = self.text.len; self.anchor = self.caret; }
    __w_mark(self);
    return __w_self(self);
}
double Widget::value() const { return self.sliderValue; }
&Widget Widget::set_range(double minValue, double maxValue) {
    self.sliderMin = minValue;
    self.sliderMax = maxValue;
    return self.set_value(self.sliderValue);
}
&Widget Widget::set_step(double step) { self.step = step < 0.0 ? 0.0 : step; return self.set_value(self.sliderValue); }
&Widget Widget::set_decimals(int decimals) { self.decimals = decimals < 0 ? 0 : decimals; return self.set_value(self.sliderValue); }
&Widget Widget::set_indeterminate(int on) { self.indeterminate = on; __w_mark(self); return __w_self(self); }
&Widget Widget::set_checked(int checked) {
    self.checked = checked ? 1 : 0;
    if (self.kind == WIDGET_MENUITEM) { self.wrap = 1; }   // a checkable menu item from now on
    __w_mark(self);
    return __w_self(self);
}
int Widget::is_checked() const { return self.checked; }

&Widget Widget::add_item(const &char text) {
    self.items.push(String::from(text));
    if (self.kind == WIDGET_COMBO && self.selected < 0) { self.selected = 0; }
    __w_mark(self);
    return __w_self(self);
}
void Widget::clear_items() {
    unsigned long i = 0UL;
    while (i < self.items.len) { self.items.at(i).free(); i = i + 1UL; }
    self.items.clear();
    self.selected = -1;
    self.hotIndex = -1;
    __w_mark(self);
}
unsigned long Widget::item_count() const { return self.items.len; }
const &char Widget::item_text(unsigned long i) const {
    if (i >= self.items.len) { return ""; }
    unsafe { return (const char*)self.items.at(i).data; }
}
&Widget Widget::set_selected(int index) {
    int n = (int)self.items.len;
    if (self.kind == WIDGET_TABS) { n = (int)self.children.len; }
    else if (self.kind == WIDGET_TABLE) { n = self.rowCount; }
    else if (self.kind == WIDGET_TREE) { n = (int)self.nodes.len; }
    if (index < -1) { index = -1; }
    if (index >= n) { index = n - 1; }
    self.selected = index;
    __w_mark(self);
    return __w_self(self);
}
int Widget::selected_index() const { return self.selected; }

&Widget Widget::add_tab(const &char title, &Widget page) {
    self.items.push(String::from(title));
    self.add_child(page);
    if (self.selected < 0) { self.selected = 0; }
    return __w_self(self);
}

int Widget::add_node(int parent, const &char text) {
    struct WidgetTreeNode n;
    n.text = String::from(text);
    n.parent = parent;
    n.expanded = 0;
    n.depth = 0;
    n.tag = 0L;
    if (parent >= 0 && (unsigned long)parent < self.nodes.len) { n.depth = self.nodes.at((unsigned long)parent).depth + 1; }
    else { n.parent = -1; }
    self.nodes.push(n);
    __w_mark(self);
    return (int)self.nodes.len - 1;
}
&Widget Widget::set_expanded(int node, int expanded) {
    if (node >= 0 && (unsigned long)node < self.nodes.len) {
        &WidgetTreeNode n = self.nodes.at((unsigned long)node);
        n.expanded = expanded;
    }
    __w_mark(self);
    return __w_self(self);
}
int Widget::node_count() const { return (int)self.nodes.len; }
const &char Widget::node_text(int node) const {
    if (node < 0 || (unsigned long)node >= self.nodes.len) { return ""; }
    unsafe { return (const char*)self.nodes.at((unsigned long)node).text.data; }
}

&Widget Widget::add_column(const &char title, int width) {
    // Existing rows get an empty cell for the new column.
    unsigned long oldCols = self.tableCols.len;
    if (self.rowCount > 0 && oldCols > 0UL) {
        struct Vec<struct String> cells = Vec::new();
        int r = 0;
        while (r < self.rowCount) {
            unsigned long c = 0UL;
            while (c < oldCols) { cells.push(*self.cells.at((unsigned long)r * oldCols + c)); c = c + 1UL; }
            cells.push(String::new());
            r = r + 1;
        }
        self.cells.free();
        self.cells = cells;
    }
    struct WidgetColumn col;
    col.title = String::from(title);
    col.width = width;
    col.align = GUI_TEXT_LEFT;
    self.tableCols.push(col);
    __w_mark(self);
    return __w_self(self);
}
int Widget::add_row() {
    unsigned long c = 0UL;
    while (c < self.tableCols.len) { self.cells.push(String::new()); c = c + 1UL; }
    self.rowCount = self.rowCount + 1;
    __w_mark(self);
    return self.rowCount - 1;
}
&Widget Widget::set_cell_text(int row, int col, const &char text) {
    if (row < 0 || row >= self.rowCount || col < 0 || (unsigned long)col >= self.tableCols.len) { return __w_self(self); }
    &String s = self.cells.at((unsigned long)row * self.tableCols.len + (unsigned long)col);
    s.free();
    *s = String::from(text);
    __w_mark(self);
    return __w_self(self);
}
const &char Widget::cell_text(int row, int col) const {
    if (row < 0 || row >= self.rowCount || col < 0 || (unsigned long)col >= self.tableCols.len) { return ""; }
    unsafe { return (const char*)self.cells.at((unsigned long)row * self.tableCols.len + (unsigned long)col).data; }
}
int Widget::row_count() const { return self.rowCount; }
void Widget::clear_rows() {
    unsigned long i = 0UL;
    while (i < self.cells.len) { self.cells.at(i).free(); i = i + 1UL; }
    self.cells.clear();
    self.rowCount = 0;
    self.selected = -1;
    __w_mark(self);
}

&Widget Widget::select(unsigned long start, unsigned long end) {
    if (start > self.text.len) { start = self.text.len; }
    if (end > self.text.len) { end = self.text.len; }
    self.anchor = start;
    self.caret = end;
    __w_mark(self);
    return __w_self(self);
}
&Widget Widget::select_all() { return self.select(0UL, self.text.len); }
unsigned long Widget::selection_start() const { return self.anchor < self.caret ? self.anchor : self.caret; }
unsigned long Widget::selection_end() const { return self.anchor < self.caret ? self.caret : self.anchor; }

&Widget Widget::set_image(const ?&GuiImage img) { self.image = img; __w_mark(self); return __w_self(self); }
&Widget Widget::set_fit(int fit) { self.fit = fit; __w_mark(self); return __w_self(self); }

&Widget Widget::set_color(struct GuiColor c) {
    self.color = c;
    float h = 0.0f; float s = 0.0f; float v = 0.0f;
    c.to_hsv(&h, &s, &v);
    // Keep the hue when the color is grey (undefined hue).
    if (s > 0.0f && v > 0.0f) { self.hue = h; }
    self.sat = s;
    self.val = v;
    __w_mark(self);
    return __w_self(self);
}
struct GuiColor Widget::get_color() const { return self.color; }

&Widget Widget::set_shortcut(const &char shortcut) {
    self.shortcut.free();
    self.shortcut = String::from(shortcut);
    return __w_self(self);
}

&Widget Widget::on_click_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData) {
    unsafe { self.onClick = cb; self.onClickData = userData; }
    return __w_self(self);
}
&Widget Widget::on_change_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData) {
    unsafe { self.onChange = cb; self.onChangeData = userData; }
    return __w_self(self);
}
&Widget Widget::on_activate_raw_(fn void(&Widget, ?&Erased) cb, ?&Erased userData) {
    unsafe { self.onActivate = cb; self.onActivateData = userData; }
    return __w_self(self);
}
&Widget Widget::on_draw_raw_(fn void(&Widget, &GuiCanvas, ?&Erased) cb, ?&Erased userData) {
    unsafe { self.onDraw = cb; self.onDrawData = userData; }
    return __w_self(self);
}

void __w_fire_click(&Widget w) {
    unsafe {
        if (!w.onClick.is_null()) {
            WidgetCallback cb = w.onClick!;
            cb(w, w.onClickData);
        }
    }
}
void __w_fire_change(&Widget w) {
    __w_mark(w);
    unsafe {
        if (!w.onChange.is_null()) {
            WidgetCallback cb = w.onChange!;
            cb(w, w.onChangeData);
        }
    }
}
void __w_fire_activate(&Widget w) {
    unsafe {
        if (!w.onActivate.is_null()) {
            WidgetCallback cb = w.onActivate!;
            cb(w, w.onActivateData);
        }
    }
}

// ── metrics and drawing helpers ───────────────────────────────────────────

float __w_px(const &GuiUi ui, float logical) { return logical * ui.scale; }
int __w_ipx(const &GuiUi ui, int logical) { return (int)((float)logical * ui.scale + 0.5f); }

float __w_font_size(const &GuiUi ui, const &Widget w) {
    float s = w.style.fontSize > 0.0f ? w.style.fontSize : ui.theme.fontSize;
    if (w.style.fontScale > 1 && w.style.font.is_null()) { s = s * (float)w.style.fontScale; }
    return s * ui.scale;
}

&GuiTypeface __w_face(const &GuiUi ui, const &Widget w) {
    if (w.style.bold) {
        match (ui.theme.boldFont) {
            case some(f): { unsafe { return (struct GuiTypeface*)f; } }
            case null: { }
        }
    }
    match (ui.theme.font) {
        case some(f): { unsafe { return (struct GuiTypeface*)f; } }
        case null: { }
    }
    return GuiTypeface::fallback();
}

float __w_line_h(const &GuiUi ui, const &Widget w) {
    match (w.style.font) {
        case some(bf): { return (float)bf.text_height(w.style.fontScale) * ui.scale; }
        case null: { }
    }
    &GuiTypeface f = __w_face(ui, w);
    return f.line_height(__w_font_size(ui, w));
}

float __w_ascent(const &GuiUi ui, const &Widget w) {
    match (w.style.font) {
        case some(bf): { return (float)bf.text_height(w.style.fontScale) * ui.scale; }
        case null: { }
    }
    &GuiTypeface f = __w_face(ui, w);
    float sz = __w_font_size(ui, w);
    return f.ascent(sz) + f.line_gap(sz) * 0.5f;
}

float __w_text_wn(const &GuiUi ui, const &Widget w, const &char s, unsigned long n) {
    match (w.style.font) {
        case some(bf): {
            // bitmap font: one cell per byte of ASCII
            return (float)(bf.glyphW * w.style.fontScale) * (float)n * ui.scale;
        }
        case null: { }
    }
    &GuiTypeface f = __w_face(ui, w);
    return f.measure_n_n_(s, n, __w_font_size(ui, w));
}

float __w_text_w(const &GuiUi ui, const &Widget w, const &char s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != '\0') { n = n + 1UL; } }
    return __w_text_wn(ui, w, s, n);
}

void __w_text_n(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, unsigned long n,
                float x, float y, struct GuiColor col) {
    match (w.style.font) {
        case some(bf): {
            // A bitmap GuiFont: its pixels as little squares, through the canvas.
            float cell = (float)w.style.fontScale * ui.scale;
            c.set_fill_color(col);
            unsigned long i = 0UL;
            while (i < n) {
                int cp;
                unsafe { cp = (int)(unsigned char)s[i]; }
                int py = 0;
                while (py < bf.glyphH) {
                    int px = 0;
                    while (px < bf.glyphW) {
                        if (bf.get_bit(cp, px, py)) {
                            c.fill_rect(x + ((float)((int)i * bf.glyphW + px)) * cell, y + (float)py * cell, cell, cell);
                        }
                        px = px + 1;
                    }
                    py = py + 1;
                }
                i = i + 1UL;
            }
            return;
        }
        case null: { }
    }
    &GuiTypeface f = __w_face(ui, w);
    c.set_font(f, __w_font_size(ui, w));
    c.set_text_align(GUI_TEXT_LEFT);
    c.set_text_baseline(GUI_BASELINE_ALPHABETIC);
    c.set_fill_color(col);
    c.fill_text_n_n_(s, n, x, floor_f(y + __w_ascent(ui, w) + 0.5f));
}

void __w_text(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, float x, float y, struct GuiColor col) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != '\0') { n = n + 1UL; } }
    __w_text_n(ui, c, w, s, n, x, y, col);
}

void __w_text_in(const &GuiUi ui, &GuiCanvas c, const &Widget w, const &char s, float x, float y, float bw, float bh,
                 int align, struct GuiColor col) {
    float tw = __w_text_w(ui, w, s);
    float th = __w_line_h(ui, w);
    float tx = x;
    if (align == GUI_TEXT_CENTER) { tx = x + (bw - tw) * 0.5f; }
    else if (align == GUI_TEXT_RIGHT) { tx = x + bw - tw; }
    float ty = y + (bh - th) * 0.5f;
    __w_text(ui, c, w, s, floor_f(tx + 0.5f), ty, col);
}

static int __w_kind_padding(const &GuiUi ui, int kind, int side) {
    int p = ui.theme.padding;
    if (kind == WIDGET_BUTTON || kind == WIDGET_COMBO) { return (side == 0 || side == 2) ? p * 2 : p / 2; }
    if (kind == WIDGET_TEXTINPUT || kind == WIDGET_SPINNER || kind == WIDGET_TEXTAREA) { return (side == 0 || side == 2) ? p : p / 2; }
    if (kind == WIDGET_GROUP || kind == WIDGET_PANEL || kind == WIDGET_DIALOG) { return p + 4; }
    if (kind == WIDGET_STATUSBAR) { return (side == 0 || side == 2) ? p : 2; }
    if (kind == WIDGET_MENUITEM) { return (side == 0 || side == 2) ? p + 2 : 0; }
    if (kind == WIDGET_TOOLTIP) { return (side == 0 || side == 2) ? p : p / 2; }
    return 0;
}

int __w_pad(const &GuiUi ui, const &Widget w, int side) {
    int v = side == 0 ? w.padL : (side == 1 ? w.padT : (side == 2 ? w.padR : w.padB));
    if (v >= 0) { return __w_ipx(ui, v); }
    if ((w.style.setMask & WIDGET_STYLE_PADDING) != 0) { return __w_ipx(ui, w.style.padding); }
    return __w_ipx(ui, __w_kind_padding(ui, w.kind, side));
}

int __w_spacing(const &GuiUi ui, const &Widget w) {
    if ((w.style.setMask & WIDGET_STYLE_SPACING) != 0) { return __w_ipx(ui, w.style.spacing); }
    if (w.kind == WIDGET_STATUSBAR || w.kind == WIDGET_MENUBAR || w.kind == WIDGET_MENU) { return 0; }
    return __w_ipx(ui, ui.theme.spacing);
}

struct GuiColor __w_fg(const &GuiUi ui, const &Widget w) {
    if (!__w_effectively_enabled(w)) { return ui.theme.textDisabled; }
    if ((w.style.setMask & WIDGET_STYLE_FG) != 0) { return w.style.fg; }
    return ui.theme.text;
}

struct GuiColor __w_bg_or(const &Widget w, struct GuiColor themeColor) {
    if ((w.style.setMask & WIDGET_STYLE_BG) != 0) { return w.style.bg; }
    return themeColor;
}

struct GuiColor __w_border_or(const &Widget w, struct GuiColor themeColor) {
    if ((w.style.setMask & WIDGET_STYLE_BORDER) != 0) { return w.style.border; }
    return themeColor;
}

void __w_ring(&GuiCanvas c, float x, float y, float w, float h, float r, float bw, struct GuiColor col) {
    if (bw <= 0.0f || col.a == (unsigned char)0 || w <= 0.0f || h <= 0.0f) { return; }
    // The border as a filled ring (outer minus inner, even-odd): cheaper
    // than stroking and exactly bw wide.
    c.scratchPath.clear();
    c.scratchPath.rounded_rect(x, y, w, h, r);
    float ir = r - bw;
    if (ir < 0.0f) { ir = 0.0f; }
    if (w > 2.0f * bw && h > 2.0f * bw) { c.scratchPath.rounded_rect(x + bw, y + bw, w - 2.0f * bw, h - 2.0f * bw, ir); }
    int rule = c.st.fillRule;
    c.set_fill_rule(GUI_FILL_EVENODD);
    c.set_fill_color(col);
    c.fill_path(&c.scratchPath);
    c.set_fill_rule(rule);
}

void __w_box(&GuiCanvas c, float x, float y, float w, float h, float r, struct GuiColor fill,
             struct GuiColor border, float bw) {
    if (w <= 0.0f || h <= 0.0f) { return; }
    if (fill.a != (unsigned char)0) {
        c.set_fill_color(fill);
        if (r > 0.0f) { c.fill_rounded_rect(x, y, w, h, r); } else { c.fill_rect(x, y, w, h); }
    }
    __w_ring(c, x, y, w, h, r, bw, border);
}

void __w_focus_ring(const &GuiUi ui, &GuiCanvas c, const &Widget w, float x, float y, float bw, float bh, float r) {
    float t = __w_px(ui, 2.0f);
    __w_ring(c, x - t * 0.5f, y - t * 0.5f, bw + t, bh + t, r + t * 0.5f, t, ui.theme.accent.with_alpha(0.75f));
}

void __w_chevron(&GuiCanvas c, float cx, float cy, float size, int dir, struct GuiColor col, float lw) {
    float h = size * 0.5f;
    float q = size * 0.25f;
    c.save();
    c.set_stroke_color(col);
    c.set_line_width(lw);
    c.set_line_cap(GUI_CAP_ROUND);
    c.set_line_join(GUI_JOIN_ROUND);
    float noDash[1];
    noDash[0] = 0.0f;
    c.set_dash_n_(&noDash[0], 0, 0.0f);
    c.begin_path();
    if (dir == 0) { c.move_to(cx - h, cy - q); c.line_to(cx, cy + q); c.line_to(cx + h, cy - q); }
    else if (dir == 1) { c.move_to(cx - q, cy - h); c.line_to(cx + q, cy); c.line_to(cx - q, cy + h); }
    else if (dir == 2) { c.move_to(cx - h, cy + q); c.line_to(cx, cy - q); c.line_to(cx + h, cy + q); }
    else { c.move_to(cx + q, cy - h); c.line_to(cx - q, cy); c.line_to(cx + q, cy + h); }
    c.stroke();
    c.restore();
}

void __w_check_mark(&GuiCanvas c, float x, float y, float s, struct GuiColor col, float lw) {
    c.save();
    c.set_stroke_color(col);
    c.set_line_width(lw);
    c.set_line_cap(GUI_CAP_ROUND);
    c.set_line_join(GUI_JOIN_ROUND);
    float noDash[1];
    noDash[0] = 0.0f;
    c.set_dash_n_(&noDash[0], 0, 0.0f);
    c.begin_path();
    c.move_to(x + s * 0.22f, y + s * 0.52f);
    c.line_to(x + s * 0.42f, y + s * 0.72f);
    c.line_to(x + s * 0.78f, y + s * 0.30f);
    c.stroke();
    c.restore();
}

// ── focus and enablement ──────────────────────────────────────────────────

int __w_effectively_enabled(const &Widget w) {
    const ?&Widget cur = w;
    int guard = 0;
    while (guard < 10000) {
        match (cur) {
            case null: return 1;
            case some(x): {
                if (!x.enabled) {
                    // a menu separator is "disabled" by design but its menu isn't
                    return 0;
                }
                cur = x.parent;
            }
        }
        guard = guard + 1;
    }
    return 1;
}

int __w_focusable(const &Widget w) {
    if (w.focusable >= 0) { return w.focusable; }
    int k = w.kind;
    return (k == WIDGET_BUTTON || k == WIDGET_CHECKBOX || k == WIDGET_TOGGLE || k == WIDGET_RADIO ||
            k == WIDGET_SLIDER || k == WIDGET_SPINNER || k == WIDGET_TEXTINPUT || k == WIDGET_TEXTAREA ||
            k == WIDGET_COMBO || k == WIDGET_LIST || k == WIDGET_TREE || k == WIDGET_TABLE ||
            k == WIDGET_TABS || k == WIDGET_COLOR) ? 1 : 0;
}

void __w_set_cursor(&GuiUi ui, int cursor) {
    ui.cursor = cursor;
    match (ui.win) {
        case some(win): { win.set_cursor(cursor); }
        case null: { }
    }
}

// ── measuring and layout ──────────────────────────────────────────────────

static int __w_family(int kind) {
    // 0 basic, 1 text, 2 list, 3 layout, 4 menu, 5 custom/spacer
    if (kind == WIDGET_TEXTINPUT || kind == WIDGET_TEXTAREA) { return 1; }
    if (kind == WIDGET_COMBO || kind == WIDGET_LIST || kind == WIDGET_TREE || kind == WIDGET_TABLE ||
        kind == WIDGET_POPUPLIST) { return 2; }
    if (kind == WIDGET_VSTACK || kind == WIDGET_HSTACK || kind == WIDGET_GRID || kind == WIDGET_STACK ||
        kind == WIDGET_SCROLL || kind == WIDGET_SPLITTER || kind == WIDGET_TABS || kind == WIDGET_GROUP ||
        kind == WIDGET_PANEL || kind == WIDGET_STATUSBAR || kind == WIDGET_DIALOG) { return 3; }
    if (kind == WIDGET_MENUBAR || kind == WIDGET_MENU || kind == WIDGET_MENUITEM || kind == WIDGET_TOOLTIP) { return 4; }
    if (kind == WIDGET_CUSTOM || kind == WIDGET_SPACER) { return 5; }
    return 0;
}

void __w_measure(&GuiUi ui, &Widget w, &int mw, &int mh) {
    *mw = 0;
    *mh = 0;
    if (!w.visible) { return; }
    int cw = 0;
    int ch = 0;
    int fam = __w_family(w.kind);
    if (fam == 0) { __wk_basic_measure(ui, w, &cw, &ch); }
    else if (fam == 1) { __wk_text_measure(ui, w, &cw, &ch); }
    else if (fam == 2) { __wk_list_measure(ui, w, &cw, &ch); }
    else if (fam == 3) { __wk_layout_measure(ui, w, &cw, &ch); }
    else if (fam == 4) { __wk_menu_measure(ui, w, &cw, &ch); }
    int ow = cw + __w_pad(ui, w, 0) + __w_pad(ui, w, 2);
    int oh = ch + __w_pad(ui, w, 1) + __w_pad(ui, w, 3);
    if (w.prefW > 0) { ow = __w_ipx(ui, w.prefW); }
    if (w.prefH > 0) { oh = __w_ipx(ui, w.prefH); }
    if (w.minW > 0 && ow < __w_ipx(ui, w.minW)) { ow = __w_ipx(ui, w.minW); }
    if (w.minH > 0 && oh < __w_ipx(ui, w.minH)) { oh = __w_ipx(ui, w.minH); }
    if (w.maxW > 0 && ow > __w_ipx(ui, w.maxW)) { ow = __w_ipx(ui, w.maxW); }
    if (w.maxH > 0 && oh > __w_ipx(ui, w.maxH)) { oh = __w_ipx(ui, w.maxH); }
    *mw = ow;
    *mh = oh;
}

int __w_height_for(&GuiUi ui, &Widget w, int width) {
    int mw = 0; int mh = 0;
    __w_measure(ui, w, &mw, &mh);
    if (!w.visible || w.prefH > 0) { return mh; }
    if (w.kind == WIDGET_LABEL && w.wrap) {
        int inner = width - __w_pad(ui, w, 0) - __w_pad(ui, w, 2);
        if (inner < 1) { inner = 1; }
        struct GuiTextLayout l = GuiTextLayout::new();
        unsafe { l.build_n_(__w_face(ui, w), __w_font_size(ui, w), (const char*)w.text.data, w.text.len, (float)inner, w.textAlign); }
        int h = (int)ceil_f(l.height) + __w_pad(ui, w, 1) + __w_pad(ui, w, 3);
        l.free();
        if (w.minH > 0 && h < __w_ipx(ui, w.minH)) { h = __w_ipx(ui, w.minH); }
        return h;
    }
    if ((w.kind == WIDGET_VSTACK || w.kind == WIDGET_GROUP || w.kind == WIDGET_PANEL) && w.children.len > 0UL) {
        // a column: the sum of its children's heights at this width
        int inner = width - __w_pad(ui, w, 0) - __w_pad(ui, w, 2);
        int total = 0;
        int n = 0;
        unsigned long i = 0UL;
        while (i < w.children.len) {
            &Widget c = *w.children.at(i);
            i = i + 1UL;
            if (!c.visible) { continue; }
            int cw = inner - __w_ipx(ui, c.marginL) - __w_ipx(ui, c.marginR);
            total = total + __w_height_for(ui, c, cw) + __w_ipx(ui, c.marginT) + __w_ipx(ui, c.marginB);
            n = n + 1;
        }
        if (n > 1) { total = total + __w_spacing(ui, w) * (n - 1); }
        int h = total + __w_pad(ui, w, 1) + __w_pad(ui, w, 3);
        if (w.kind == WIDGET_GROUP) { h = h + (int)__w_line_h(ui, w); }
        return h > mh ? h : mh;
    }
    return mh;
}

void __w_content(const &GuiUi ui, const &Widget w, &int cx, &int cy, &int cw, &int ch) {
    int l = __w_pad(ui, w, 0);
    int t = __w_pad(ui, w, 1);
    int r = __w_pad(ui, w, 2);
    int b = __w_pad(ui, w, 3);
    *cx = w.x + l;
    *cy = w.y + t;
    *cw = w.w - l - r;
    *ch = w.h - t - b;
    if (*cw < 0) { *cw = 0; }
    if (*ch < 0) { *ch = 0; }
}

void __w_arrange(&GuiUi ui, &Widget w, int x, int y, int ww, int hh) {
    if (w.kind == WIDGET_CUSTOM) {
        unsafe {
            if (!w.customLayout.is_null()) {
                WidgetLayoutFn fnp = w.customLayout!;
                fnp(w, x, y, ww, hh);
                return;
            }
        }
    }
    w.x = x; w.y = y; w.w = ww < 0 ? 0 : ww; w.h = hh < 0 ? 0 : hh;
    if (!w.visible) { return; }
    int fam = __w_family(w.kind);
    if (fam == 3) { __wk_layout_arrange(ui, w); }
    else if (fam == 4) { __wk_menu_arrange(ui, w); }
    else if (fam == 0) { __wk_basic_arrange(ui, w); }
    else if (fam == 5 && w.children.len > 0UL) {
        // custom/spacer with children: stack them in the content box
        int cx = 0; int cy = 0; int cw = 0; int ch = 0;
        __w_content(ui, w, &cx, &cy, &cw, &ch);
        unsigned long i = 0UL;
        while (i < w.children.len) { __w_arrange(ui, *w.children.at(i), cx, cy, cw, ch); i = i + 1UL; }
    }
}

// ── the UI context ────────────────────────────────────────────────────────

static &GuiUi __w_ui_of(&Widget root) {
    ?&GuiUi u = root.ui();
    match (u) {
        case some(ui): return ui;
        case null: { }
    }
    return GuiUi::new(root);
}

&GuiUi GuiUi::new(&Widget root) {
    &GuiUi ui;
    unsafe {
        ui = (struct GuiUi*)alloc_zeroed_raw_(sizeof(struct GuiUi));
        root.uiPtr = ui;
    }
    ui.root = root;
    ui.win = null;
    ui.theme = GuiTheme::light();
    ui.scale = 1.0f;
    ui.canvasReady = 0;
    ui.overlays = Vec::new();
    ui.focus = null;
    ui.hoverW = null;
    ui.capture = null;
    ui.mouseX = -1.0f; ui.mouseY = -1.0f;
    ui.mouseInside = 0;
    ui.tipFor = null;
    ui.tipSince = 0.0;
    ui.tip = __widget_new(WIDGET_TOOLTIP);
    ui.tip.uiPtr = ui;
    ui.tipShown = 0;
    ui.nowMs = 0.0;
    ui.needsLayout = 1;
    ui.dirty = 1;
    ui.layoutW = -1; ui.layoutH = -1;
    ui.cursor = GUI_CURSOR_ARROW;
    return ui;
}

void GuiUi::attach(&GuiWindow win) {
    self.win = win;
    float s = (float)win.scale;
    if (s <= 0.0f) { s = 1.0f; }
    if (s != self.scale) { self.scale = s; self.needsLayout = 1; }
    self.dirty = 1;
}

void GuiUi::set_theme(struct GuiTheme t) {
    self.theme = t;
    self.needsLayout = 1;
    self.dirty = 1;
}

void GuiUi::invalidate() { self.dirty = 1; }
void GuiUi::request_layout() { self.needsLayout = 1; self.dirty = 1; }
int GuiUi::needs_redraw() const { return (self.dirty || self.needsLayout) ? 1 : 0; }

static int __w_in(const &Widget a, const &Widget sub) {
    // 1 if 'a' is 'sub' or one of its ancestors
    const ?&Widget cur = sub;
    int guard = 0;
    while (guard < 10000) {
        match (cur) {
            case null: return 0;
            case some(x): {
                if (__w_same(x, a)) { return 1; }
                cur = x.parent;
            }
        }
        guard = guard + 1;
    }
    return 0;
}

static void __w_clear_flags(&Widget w) {
    w.hover = 0;
    w.pressed = 0;
    w.focused = 0;
    unsigned long i = 0UL;
    while (i < w.children.len) { __w_clear_flags(*w.children.at(i)); i = i + 1UL; }
}

static void __w_ui_forget(&Widget gone) {
    ?&GuiUi u = gone.ui();
    match (u) {
        case null: return;
        case some(ui): {
            match (ui.focus) { case some(f): { if (__w_in(gone, f)) { f.focused = 0; ui.focus = null; } } case null: { } }
            match (ui.hoverW) { case some(f): { if (__w_in(gone, f)) { ui.hoverW = null; } } case null: { } }
            match (ui.capture) { case some(f): { if (__w_in(gone, f)) { ui.capture = null; } } case null: { } }
            match (ui.tipFor) { case some(f): { if (__w_in(gone, f)) { ui.tipFor = null; ui.tipShown = 0; } } case null: { } }
            ui.needsLayout = 1;
            ui.dirty = 1;
        }
    }
}

void GuiUi::set_focus(?&Widget w) {
    match (self.focus) {
        case some(old): {
            int same = 0;
            unsafe { same = ((void*)old == (void*)w) ? 1 : 0; }
            if (!same) {
                old.focused = 0;
                if (old.kind == WIDGET_SPINNER) { __w_spinner_commit(old); }
                if (old.kind == WIDGET_TEXTINPUT || old.kind == WIDGET_TEXTAREA) { old.anchor = old.caret; }
            }
        }
        case null: { }
    }
    self.focus = w;
    match (w) {
        case some(nw): { nw.focused = 1; }
        case null: { }
    }
    self.dirty = 1;
}

?&Widget GuiUi::modal() const {
    unsigned long i = self.overlays.len;
    while (i > 0UL) {
        i = i - 1UL;
        &Widget o = *self.overlays.at(i);
        if (o.kind == WIDGET_DIALOG) { return o; }
    }
    return null;
}

int __w_is_popup(const &Widget w) {
    return (w.kind == WIDGET_MENU || w.kind == WIDGET_POPUPLIST) ? 1 : 0;
}

static void __w_ui_win_size(const &GuiUi ui, &int ww, &int wh) {
    *ww = ui.layoutW > 0 ? ui.layoutW : ui.root.w;
    *wh = ui.layoutH > 0 ? ui.layoutH : ui.root.h;
    match (ui.win) {
        case some(win): { *ww = win.width; *wh = win.height; }
        case null: { }
    }
}

void GuiUi::popup(&Widget menu, float x, float y) {
    unsafe { menu.uiPtr = (?&GuiUi)(void*)self; }
    menu.open = 1;
    menu.hotIndex = -1;
    menu.parent = null;
    int mw = 0; int mh = 0;
    __w_measure(self, menu, &mw, &mh);
    int ww = 0; int wh = 0;
    __w_ui_win_size(self, &ww, &wh);
    int px = (int)x;
    int py = (int)y;
    if (px + mw > ww) { px = ww - mw; }
    if (py + mh > wh) {
        // flip above the anchor point if that fits better
        if ((int)y - mh >= 0) { py = (int)y - mh; } else { py = wh - mh; }
    }
    if (px < 0) { px = 0; }
    if (py < 0) { py = 0; }
    __w_arrange(self, menu, px, py, mw, mh);
    self.overlays.push(menu);
    self.dirty = 1;
}

void __w_popup_close_from(&GuiUi ui, &Widget popup) {
    unsigned long i = 0UL;
    while (i < ui.overlays.len) {
        if (__w_same(*ui.overlays.at(i), popup)) { break; }
        i = i + 1UL;
    }
    if (i >= ui.overlays.len) { return; }
    while (ui.overlays.len > i) {
        &Widget o = *ui.overlays.at(ui.overlays.len - 1UL);
        if (o.kind == WIDGET_DIALOG && !__w_same(o, popup)) { break; }
        o.open = 0;
        o.hotIndex = -1;
        match (o.owner) { case some(ow): { ow.open = 0; ow.popup = null; } case null: { } }
        match (ui.focus) { case some(f): { if (__w_in(o, f)) { ui.set_focus(null); } } case null: { } }
        ui.overlays.pop();
    }
    ui.dirty = 1;
}

void GuiUi::close_popups() {
    // every popup above the topmost dialog
    while (self.overlays.len > 0UL) {
        &Widget o = *self.overlays.at(self.overlays.len - 1UL);
        if (o.kind == WIDGET_DIALOG) { break; }
        __w_popup_close_from(self, o);
    }
}

static ?&Widget __w_first_focusable(&Widget w);

void GuiUi::show_modal(&Widget dialog) {
    self.close_popups();
    unsafe { dialog.uiPtr = (?&GuiUi)(void*)self; }
    dialog.open = 1;
    dialog.result = GUI_RESULT_NONE;
    self.overlays.push(dialog);
    int mw = 0; int mh = 0;
    __w_measure(self, dialog, &mw, &mh);
    int ww = 0; int wh = 0;
    __w_ui_win_size(self, &ww, &wh);
    if (mw > ww) { mw = ww; }
    if (mh > wh) { mh = wh; }
    __w_arrange(self, dialog, (ww - mw) / 2, (wh - mh) / 3, mw, mh);
    dialog.dragX = 0.0f;
    self.set_focus(__w_first_focusable(dialog));
    self.dirty = 1;
}

void GuiUi::close_modal(&Widget dialog) {
    unsigned long i = 0UL;
    while (i < self.overlays.len) {
        if (__w_same(*self.overlays.at(i), dialog)) {
            // close what's above it first (its popups)
            while (self.overlays.len > i + 1UL) {
                &Widget o = *self.overlays.at(self.overlays.len - 1UL);
                o.open = 0;
                self.overlays.pop();
            }
            self.overlays.remove(i);
            dialog.open = 0;
            match (self.focus) { case some(f): { if (__w_in(dialog, f)) { self.set_focus(null); } } case null: { } }
            match (self.capture) { case some(f): { if (__w_in(dialog, f)) { self.capture = null; } } case null: { } }
            match (self.hoverW) { case some(f): { if (__w_in(dialog, f)) { self.hoverW = null; } } case null: { } }
            self.dirty = 1;
            return;
        }
        i = i + 1UL;
    }
}

void Widget::close_dialog(int result) {
    self.result = result;
    ?&GuiUi u = self.ui();
    // the dialog itself or the dialog this widget is in
    ?&Widget cur = self;
    int guard = 0;
    while (guard < 1000) {
        match (cur) {
            case null: return;
            case some(d): {
                if (d.kind == WIDGET_DIALOG) {
                    d.result = result;
                    match (u) { case some(ui): { ui.close_modal(d); } case null: { } }
                    // the message-box close callback
                    match (d.onClose_) {
                        case some(cb): cb(d, result, d.onCloseData_);
                        case null: { }
                    }
                    return;
                }
                cur = d.parent;
            }
        }
        guard = guard + 1;
    }
}

// ── painting ──────────────────────────────────────────────────────────────

void __w_paint_children(&GuiUi ui, &GuiCanvas c, &Widget w) {
    unsigned long i = 0UL;
    while (i < w.children.len) {
        __w_paint(ui, c, *w.children.at(i));
        i = i + 1UL;
    }
}

void __w_paint(&GuiUi ui, &GuiCanvas c, &Widget w) {
    if (!w.visible || w.w <= 0 || w.h <= 0) { return; }
    // Off the clip: nothing to do.
    struct GuiRect cb = c.clip_bounds();
    float ox = c.st.m.e;
    float oy = c.st.m.f;
    if ((float)(w.x + w.w) + ox <= cb.x || (float)w.x + ox >= cb.x + cb.w ||
        (float)(w.y + w.h) + oy <= cb.y || (float)w.y + oy >= cb.y + cb.h) { return; }
    int fam = __w_family(w.kind);
    if (w.kind == WIDGET_CUSTOM) {
        // legacy: draws into the window directly
        unsafe {
            if (!w.customDraw.is_null()) {
                match (ui.win) {
                    case some(win): {
                        WidgetDrawFn d = w.customDraw!;
                        d(win, w);
                    }
                    case null: { }
                }
            }
        }
        __w_paint_children(ui, c, w);
        return;
    }
    c.save();
    if (fam == 0) { __wk_basic_paint(ui, c, w); __w_paint_children(ui, c, w); }
    else if (fam == 1) { __wk_text_paint(ui, c, w); }
    else if (fam == 2) { __wk_list_paint(ui, c, w); }
    else if (fam == 3) {
        __wk_layout_paint(ui, c, w);   // paints and clips its children itself
    }
    else if (fam == 4) { __wk_menu_paint(ui, c, w); }
    else { __w_paint_children(ui, c, w); }
    c.restore();
}

static void __w_ensure_layout(&GuiUi ui) {
    int ww = ui.root.w;
    int wh = ui.root.h;
    match (ui.win) {
        case some(win): { ww = win.width; wh = win.height; }
        case null: { }
    }
    if (ui.needsLayout || ww != ui.layoutW || wh != ui.layoutH) {
        __w_arrange(ui, ui.root, 0, 0, ww, wh);
        ui.layoutW = ww;
        ui.layoutH = wh;
        // dialogs stay where they are, kept inside the window
        unsigned long i = 0UL;
        while (i < ui.overlays.len) {
            &Widget o = *ui.overlays.at(i);
            int mw = 0; int mh = 0;
            __w_measure(ui, o, &mw, &mh);
            int nx = o.x; int ny = o.y;
            if (nx + mw > ww) { nx = ww - mw; }
            if (ny + mh > wh) { ny = wh - mh; }
            if (nx < 0) { nx = 0; }
            if (ny < 0) { ny = 0; }
            __w_arrange(ui, o, nx, ny, mw, mh);
            i = i + 1UL;
        }
        ui.needsLayout = 0;
    }
}

void GuiUi::render() {
    match (self.win) {
        case null: return;
        case some(win): {
            float s = (float)win.scale;
            if (s <= 0.0f) { s = 1.0f; }
            if (s != self.scale) { self.scale = s; self.needsLayout = 1; }
            if (!self.canvasReady) {
                self.canvas = GuiCanvas::for_window(win);
                self.canvasReady = 1;
            } else {
                self.canvas.retarget_window(win);
            }
            self.canvas.reset();
            __w_ensure_layout(self);
            self.canvas.clear(__w_bg_or(self.root, self.theme.window));
            __w_paint(self, &self.canvas, self.root);
            unsigned long i = 0UL;
            while (i < self.overlays.len) {
                &Widget o = *self.overlays.at(i);
                if (o.kind == WIDGET_DIALOG) {
                    self.canvas.set_fill_color(self.theme.overlay);
                    self.canvas.fill_rect(0.0f, 0.0f, (float)win.width, (float)win.height);
                }
                __w_paint(self, &self.canvas, o);
                i = i + 1UL;
            }
            if (self.tipShown) { __w_paint(self, &self.canvas, self.tip); }
            self.dirty = 0;
        }
    }
}

int GuiUi::animating() const {
    // a visible busy progress bar, or a focused text field's caret
    match (self.focus) {
        case some(f): {
            if ((f.kind == WIDGET_TEXTINPUT || f.kind == WIDGET_TEXTAREA || f.kind == WIDGET_SPINNER) && !f.readOnly) { return 1; }
        }
        case null: { }
    }
    return 0;
}

// any visible indeterminate progress bar under w
static int __w_busy(const &Widget w) {
    if (!w.visible) { return 0; }
    if (w.kind == WIDGET_PROGRESS && w.indeterminate) { return 1; }
    if (w.kind == WIDGET_TABS) {
        if (w.selected >= 0 && (unsigned long)w.selected < w.children.len) { return __w_busy(*w.children.at((unsigned long)w.selected)); }
        return 0;
    }
    unsigned long i = 0UL;
    while (i < w.children.len) {
        if (__w_busy(*w.children.at(i))) { return 1; }
        i = i + 1UL;
    }
    return 0;
}

static void __w_show_tip(&GuiUi ui, &Widget target);

void GuiUi::tick(double nowMs) {
    double prev = self.nowMs;
    self.nowMs = nowMs;
    // caret blink: every 530 ms
    match (self.focus) {
        case some(f): {
            if (f.kind == WIDGET_TEXTINPUT || f.kind == WIDGET_TEXTAREA || f.kind == WIDGET_SPINNER) {
                if ((long long)(prev / 530.0) != (long long)(nowMs / 530.0)) { self.dirty = 1; }
            }
        }
        case null: { }
    }
    if (__w_busy(self.root)) { self.dirty = 1; }
    // tooltips: after 600 ms over the same widget
    if (!self.tipShown) {
        match (self.tipFor) {
            case some(t): {
                if (nowMs - self.tipSince >= 600.0 && self.capture.is_null()) { __w_show_tip(self, t); }
            }
            case null: { }
        }
    }
}

// ── hit testing ───────────────────────────────────────────────────────────

static int __w_contains(const &Widget w, float x, float y) {
    return (x >= (float)w.x && y >= (float)w.y && x < (float)(w.x + w.w) && y < (float)(w.y + w.h)) ? 1 : 0;
}

static ?&Widget __w_hit(&GuiUi ui, &Widget w, float x, float y) {
    if (!w.visible || !__w_contains(w, x, y)) { return null; }
    int fam = __w_family(w.kind);
    int descend = 1;
    if (fam == 3) { descend = __wk_layout_child_area(ui, w, x, y); }
    if (fam == 1 || fam == 2 || w.kind == WIDGET_MENUBAR || w.kind == WIDGET_MENU) { descend = 0; }
    if (descend) {
        unsigned long i = w.children.len;
        while (i > 0UL) {
            i = i - 1UL;
            &Widget c = *w.children.at(i);
            if (w.kind == WIDGET_TABS && (int)i != w.selected) { continue; }
            ?&Widget h = __w_hit(ui, c, x, y);
            if (!h.is_null()) { return h; }
        }
    }
    return w;
}

// The topmost widget at (x, y): popups and dialogs first; under a modal
// dialog nothing else is reachable (the dialog itself is returned).
static ?&Widget __w_hit_ui(&GuiUi ui, float x, float y) {
    unsigned long i = ui.overlays.len;
    while (i > 0UL) {
        i = i - 1UL;
        &Widget o = *ui.overlays.at(i);
        ?&Widget h = __w_hit(ui, o, x, y);
        if (!h.is_null()) { return h; }
        if (o.kind == WIDGET_DIALOG) { return null; }
    }
    return __w_hit(ui, ui.root, x, y);
}

// ── event routing ─────────────────────────────────────────────────────────

static int __w_handle(&GuiUi ui, &Widget w, &GuiEvent ev) {
    if (w.kind == WIDGET_CUSTOM) {
        unsafe {
            if (!w.customEvent.is_null()) {
                WidgetCustomEventFn f = w.customEvent!;
                if (f(w, ev)) { return 1; }
            }
        }
    }
    if (!__w_effectively_enabled(w)) { return 0; }
    int fam = __w_family(w.kind);
    if (fam == 0) { return __wk_basic_event(ui, w, ev); }
    if (fam == 1) { return __wk_text_event(ui, w, ev); }
    if (fam == 2) { return __wk_list_event(ui, w, ev); }
    if (fam == 3) { return __wk_layout_event(ui, w, ev); }
    if (fam == 4) { return __wk_menu_event(ui, w, ev); }
    if (w.kind == WIDGET_CUSTOM) {
        // the legacy behavior: a custom node clicks like a button
        int inside = __w_contains(w, (float)ev.x, (float)ev.y);
        if (ev.kind == GUI_EVENT_MOUSE_DOWN && inside && ev.button == GUI_BUTTON_LEFT) { w.pressed = 1; return 1; }
        if (ev.kind == GUI_EVENT_MOUSE_UP && w.pressed) {
            w.pressed = 0;
            if (inside) { __w_fire_click(w); }
            return 1;
        }
    }
    return 0;
}

// Delivers to 'target', then its ancestors, until one consumes it.
static int __w_bubble(&GuiUi ui, ?&Widget target, &GuiEvent ev) {
    ?&Widget cur = target;
    int guard = 0;
    while (guard < 10000) {
        match (cur) {
            case null: return 0;
            case some(w): {
                if (__w_handle(ui, w, ev)) { return 1; }
                if (w.parent.is_null()) { cur = w.owner; } else { cur = w.parent; }
            }
        }
        guard = guard + 1;
    }
    return 0;
}

static void __w_collect_focusable(&Widget w, &Vec<&Widget> out) {
    if (!w.visible || !w.enabled) { return; }
    if (__w_focusable(w)) { out.push(w); }
    if (w.kind == WIDGET_TABS) {
        if (w.selected >= 0 && (unsigned long)w.selected < w.children.len) {
            __w_collect_focusable(*w.children.at((unsigned long)w.selected), out);
        }
        return;
    }
    if (w.kind == WIDGET_MENU || w.kind == WIDGET_MENUBAR) { return; }
    unsigned long i = 0UL;
    while (i < w.children.len) { __w_collect_focusable(*w.children.at(i), out); i = i + 1UL; }
}

static ?&Widget __w_first_focusable(&Widget w) {
    struct Vec<&Widget> list = Vec::new();
    __w_collect_focusable(w, &list);
    ?&Widget r = null;
    if (list.len > 0UL) { r = *list.at(0UL); }
    list.free();
    return r;
}

static void __w_focus_step(&GuiUi ui, int backwards) {
    struct Vec<&Widget> list = Vec::new();
    ?&Widget scope = ui.modal();
    match (scope) {
        case some(d): { __w_collect_focusable(d, &list); }
        case null: { __w_collect_focusable(ui.root, &list); }
    }
    unsigned long n = list.len;
    if (n > 0UL) {
        unsigned long at = n;
        unsigned long i = 0UL;
        match (ui.focus) {
            case some(f): {
                while (i < n) { if (__w_same(*list.at(i), f)) { at = i; } i = i + 1UL; }
            }
            case null: { }
        }
        unsigned long next = 0UL;
        if (backwards) { next = (at == n || at == 0UL) ? n - 1UL : at - 1UL; }
        else { next = (at == n) ? 0UL : (at + 1UL) % n; }
        &Widget nw = *list.at(next);
        ui.set_focus(nw);
        if (nw.kind == WIDGET_TEXTINPUT || nw.kind == WIDGET_SPINNER) { nw.select_all(); }
    }
    list.free();
}

static void __w_show_tip(&GuiUi ui, &Widget target) {
    ui.tip.text.free();
    unsafe { ui.tip.text = String::from((const char*)target.tooltip.data); }
    int mw = 0; int mh = 0;
    __w_measure(ui, ui.tip, &mw, &mh);
    int ww = 0; int wh = 0;
    __w_ui_win_size(ui, &ww, &wh);
    int x = (int)ui.mouseX + __w_ipx(ui, 4);
    int y = (int)ui.mouseY + __w_ipx(ui, 20);
    if (x + mw > ww) { x = ww - mw; }
    if (y + mh > wh) { y = (int)ui.mouseY - mh - __w_ipx(ui, 4); }
    if (x < 0) { x = 0; }
    if (y < 0) { y = 0; }
    __w_arrange(ui, ui.tip, x, y, mw, mh);
    ui.tipShown = 1;
    ui.dirty = 1;
}

static void __w_hide_tip(&GuiUi ui) {
    if (ui.tipShown) { ui.tipShown = 0; ui.dirty = 1; }
}

static ?&Widget __w_tip_target(?&Widget w) {
    ?&Widget cur = w;
    int guard = 0;
    while (guard < 10000) {
        match (cur) {
            case null: return null;
            case some(x): {
                if (x.tooltip.len > 0UL) { return x; }
                cur = x.parent;
            }
        }
        guard = guard + 1;
    }
    return null;
}

static void __w_set_hover(&GuiUi ui, ?&Widget nh) {
    // hover flags along the path to the hovered widget
    match (ui.hoverW) {
        case some(old): {
            ?&Widget cur = old;
            int guard = 0;
            while (guard < 10000) {
                match (cur) { case null: { break; } case some(x): { x.hover = 0; x.hotIndex = x.kind == WIDGET_MENU ? x.hotIndex : -1; cur = x.parent; } }
                guard = guard + 1;
            }
        }
        case null: { }
    }
    ui.hoverW = nh;
    match (nh) {
        case some(n): {
            ?&Widget cur = n;
            int guard = 0;
            while (guard < 10000) {
                match (cur) { case null: { break; } case some(x): { x.hover = 1; cur = x.parent; } }
                guard = guard + 1;
            }
        }
        case null: { }
    }
}

static int __w_cursor_for(?&Widget w) {
    match (w) {
        case null: return GUI_CURSOR_ARROW;
        case some(x): {
            if (!__w_effectively_enabled(x)) { return GUI_CURSOR_ARROW; }
            if (x.kind == WIDGET_TEXTINPUT || x.kind == WIDGET_TEXTAREA) { return GUI_CURSOR_IBEAM; }
            if (x.kind == WIDGET_SPINNER) { return GUI_CURSOR_IBEAM; }
        }
    }
    return GUI_CURSOR_ARROW;
}

static int __w_default_key(&GuiUi ui, &GuiEvent ev) {
    int k = ev.keycode;
    if (k == GUI_KEY_TAB) {
        __w_focus_step(ui, ev.has_mod(GUI_MOD_SHIFT));
        return 1;
    }
    if (k == GUI_KEY_ESCAPE) {
        // a popup, then a dialog, then the focus
        if (ui.overlays.len > 0UL) {
            &Widget top = *ui.overlays.at(ui.overlays.len - 1UL);
            if (top.kind == WIDGET_DIALOG) { top.close_dialog(GUI_RESULT_CANCEL); }
            else { __w_popup_close_from(ui, top); }
            return 1;
        }
        ui.set_focus(null);
        return 1;
    }
    match (ui.focus) {
        case null: { }
        case some(f): {
            if (k == GUI_KEY_ENTER || k == GUI_KEY_SPACE) {
                if (f.kind == WIDGET_BUTTON) { __w_fire_click(f); return 1; }
                if (f.kind == WIDGET_CHECKBOX || f.kind == WIDGET_TOGGLE) {
                    f.checked = f.checked ? 0 : 1;
                    __w_fire_change(f);
                    __w_fire_click(f);
                    return 1;
                }
            }
            if (f.kind == WIDGET_SLIDER && (k == GUI_KEY_LEFT || k == GUI_KEY_RIGHT || k == GUI_KEY_UP || k == GUI_KEY_DOWN)) {
                double st = f.step > 0.0 ? f.step : (f.sliderMax - f.sliderMin) * 0.05;
                double v = (k == GUI_KEY_LEFT || k == GUI_KEY_DOWN) ? f.sliderValue - st : f.sliderValue + st;
                double nv = __w_clamp_value(f, v);
                if (nv != f.sliderValue) { f.sliderValue = nv; __w_fire_change(f); }
                return 1;
            }
        }
    }
    return 0;
}

int GuiUi::dispatch(&GuiEvent ev) {
    int k = ev.kind;
    int handled = 0;
    if (k == GUI_EVENT_RESIZE || k == GUI_EVENT_SCALE || k == GUI_EVENT_EXPOSE) {
        match (self.win) {
            case some(win): {
                float s = (float)win.scale;
                if (s > 0.0f && s != self.scale) { self.scale = s; }
            }
            case null: { }
        }
        self.needsLayout = 1;
        self.dirty = 1;
        self.close_popups();
        return 0;
    }
    if (k == GUI_EVENT_FOCUS_OUT || k == GUI_EVENT_MOUSE_LEAVE) {
        __w_set_hover(self, null);
        __w_hide_tip(self);
        self.tipFor = null;
        if (k == GUI_EVENT_FOCUS_OUT) {
            match (self.capture) { case some(c): { c.pressed = 0; } case null: { } }
            self.capture = null;
        }
        self.mouseInside = 0;
        self.dirty = 1;
        return 0;
    }
    if (k == GUI_EVENT_FOCUS_IN || k == GUI_EVENT_MOUSE_ENTER) { self.dirty = 1; return 0; }
    __w_ensure_layout(self);
    if (k == GUI_EVENT_MOUSE_MOVE || k == GUI_EVENT_MOUSE_DOWN || k == GUI_EVENT_MOUSE_UP || k == GUI_EVENT_SCROLL) {
        self.mouseX = (float)ev.x;
        self.mouseY = (float)ev.y;
        self.mouseInside = 1;
        ?&Widget hit = __w_hit_ui(self, (float)ev.x, (float)ev.y);
        if (k == GUI_EVENT_MOUSE_MOVE) {
            int changed = 0;
            unsafe { changed = ((void*)hit != (void*)self.hoverW) ? 1 : 0; }
            if (changed) {
                __w_set_hover(self, hit);
                self.dirty = 1;
                ?&Widget tt = __w_tip_target(hit);
                int tipChanged = 0;
                unsafe { tipChanged = ((void*)tt != (void*)self.tipFor) ? 1 : 0; }
                if (tipChanged) {
                    __w_hide_tip(self);
                    self.tipFor = tt;
                    self.tipSince = ev.timeMs > 0.0 ? ev.timeMs : self.nowMs;
                }
            }
            if (self.capture.is_null()) { __w_set_cursor(self, __w_cursor_for(hit)); }
            ?&Widget target = self.capture.is_null() ? hit : self.capture;
            handled = __w_bubble(self, target, ev);
            if (handled || !self.capture.is_null()) { self.dirty = 1; }
            return handled;
        }
        __w_hide_tip(self);
        self.tipFor = null;
        if (k == GUI_EVENT_MOUSE_DOWN) {
            // A click outside every open popup closes them (and is used up).
            if (self.overlays.len > 0UL) {
                &Widget top = *self.overlays.at(self.overlays.len - 1UL);
                if (top.kind != WIDGET_DIALOG) {
                    int inPopup = 0;
                    unsigned long i = 0UL;
                    while (i < self.overlays.len) {
                        &Widget o = *self.overlays.at(i);
                        if (o.kind != WIDGET_DIALOG && __w_contains(o, (float)ev.x, (float)ev.y)) { inPopup = 1; }
                        i = i + 1UL;
                    }
                    if (!inPopup) {
                        // clicking the menubar entry that's open toggles it closed;
                        // another entry switches menus (the menubar handles that)
                        int onBar = 0;
                        match (hit) { case some(h): { if (h.kind == WIDGET_MENUBAR) { onBar = 1; } } case null: { } }
                        self.close_popups();
                        self.dirty = 1;
                        if (!onBar) { return 1; }
                    }
                }
            }
            match (hit) {
                case null: { return 0; }
                case some(h): {
                    // right click: a context menu along the path
                    if (ev.button == GUI_BUTTON_RIGHT) {
                        ?&Widget cur = h;
                        int guard = 0;
                        while (guard < 10000) {
                            match (cur) {
                                case null: { break; }
                                case some(x): {
                                    match (x.contextMenu) {
                                        case some(m): {
                                            if (__w_effectively_enabled(x)) {
                                                self.popup(m, (float)ev.x, (float)ev.y);
                                                return 1;
                                            }
                                        }
                                        case null: { }
                                    }
                                    cur = x.parent;
                                }
                            }
                            guard = guard + 1;
                        }
                    }
                    // focus follows clicks (the nearest focusable ancestor)
                    ?&Widget cur = h;
                    int guard = 0;
                    ?&Widget newFocus = null;
                    while (guard < 10000) {
                        match (cur) {
                            case null: { break; }
                            case some(x): {
                                if (__w_focusable(x) && __w_effectively_enabled(x)) { newFocus = x; break; }
                                if (__w_is_popup(x) || x.kind == WIDGET_MENUBAR) { newFocus = self.focus; break; }
                                cur = x.parent;
                            }
                        }
                        guard = guard + 1;
                    }
                    // (a click on nothing focusable leaves the focus alone)
                    int same = 0;
                    unsafe { same = ((void*)newFocus == (void*)self.focus) ? 1 : 0; }
                    if (!same && !newFocus.is_null()) { self.set_focus(newFocus); }
                    self.capture = h;
                    handled = __w_bubble(self, h, ev);
                    self.dirty = 1;
                    return 1;
                }
            }
            return 0;
        }
        if (k == GUI_EVENT_MOUSE_UP) {
            ?&Widget target = self.capture.is_null() ? hit : self.capture;
            handled = __w_bubble(self, target, ev);
            self.capture = null;
            self.dirty = 1;
            return handled;
        }
        // scroll
        handled = __w_bubble(self, hit, ev);
        if (handled) { self.dirty = 1; }
        return handled;
    }
    if (k == GUI_EVENT_KEY_DOWN || k == GUI_EVENT_CHAR || k == GUI_EVENT_KEY_UP) {
        __w_hide_tip(self);
        // An open popup gets the keys first.
        ?&Widget target = self.focus;
        if (self.overlays.len > 0UL) {
            &Widget top = *self.overlays.at(self.overlays.len - 1UL);
            if (top.kind != WIDGET_DIALOG) { target = top; }
        }
        handled = __w_bubble(self, target, ev);
        if (!handled && k == GUI_EVENT_KEY_DOWN && !ev.defaultPrevented) {
            ?&Widget scope = self.modal();
            match (scope) {
                case some(d): { handled = __wk_menu_shortcut(self, d, ev); }
                case null: { handled = __wk_menu_shortcut(self, self.root, ev); }
            }
        }
        if (k == GUI_EVENT_KEY_DOWN && !ev.defaultPrevented) {
            // default actions also run after a widget handled the key
            if (ev.keycode == GUI_KEY_TAB || ev.keycode == GUI_KEY_ESCAPE || !handled) {
                if (__w_default_key(self, ev)) { handled = 1; }
            }
        }
        self.dirty = 1;
        return handled;
    }
    return 0;
}

void GuiUi::free() {
    self.close_popups();
    unsigned long i = 0UL;
    while (i < self.overlays.len) { &Widget o = *self.overlays.at(i); o.open = 0; i = i + 1UL; }
    self.overlays.free();
    self.tip.free();
    if (self.canvasReady) { self.canvas.free(); }
    self.root.uiPtr = null;
    self.root.free();
    unsafe { dealloc_raw_((void*)self); }
}

// ── message boxes ─────────────────────────────────────────────────────────

static void __w_mb_clicked(&Widget b, ?&Erased data) {
    b.close_dialog((int)b.tag);
}

static void __w_mb_closed(&Widget dlg, int result, ?&Erased data) {
    // the user's callback, then the box frees itself (its UI no longer
    // references it)
    match (dlg.mbUser_) {
        case some(user): user(dlg, result, dlg.mbUserData_);
        case null: { }
    }
    dlg.free();
}

void gui_message_box_raw_(&GuiUi ui, const &char title, const &char text, int buttons,
                          ?fn void(&Widget, int, ?&Erased) onClose, ?&Erased userData) {
    &Widget dlg = Widget::dialog(title);
    &Widget msg = Widget::label(text);
    msg.set_wrap(1);
    msg.set_min_size(220, 0);
    msg.set_max_size(420, 0);
    dlg.add_child(msg);
    &Widget row = Widget::hstack();
    row.add_child(Widget::spacer());
    if (buttons == 0) { buttons = GUI_MB_OK; }
    int bits[4];
    bits[0] = GUI_MB_YES; bits[1] = GUI_MB_NO; bits[2] = GUI_MB_OK; bits[3] = GUI_MB_CANCEL;
    int i = 0;
    while (i < 4) {
        if ((buttons & bits[i]) != 0) {
            const char* label = bits[i] == GUI_MB_YES ? "Yes" : (bits[i] == GUI_MB_NO ? "No" : (bits[i] == GUI_MB_OK ? "OK" : "Cancel"));
            &Widget b;
            const char* nm = bits[i] == GUI_MB_YES ? "yes" : (bits[i] == GUI_MB_NO ? "no" : (bits[i] == GUI_MB_OK ? "ok" : "cancel"));
            unsafe { b = Widget::button(label); b.set_name(nm); }
            b.set_min_size(72, 0);
            b.tag = (long)bits[i];
            b.on_click_raw_(__w_mb_clicked, null);
            row.add_child(b);
        }
        i = i + 1;
    }
    dlg.add_child(row);
    dlg.onClose_ = __w_mb_closed;
    dlg.onCloseData_ = null;
    dlg.mbUser_ = onClose;        // the user's callback (stored erased)
    dlg.mbUserData_ = userData;
    ui.show_modal(dlg);
}

// ── the original entry points ─────────────────────────────────────────────

void Widget::layout(int x, int y, int w, int h) {
    if (self.parent.is_null()) {
        &GuiUi ui = __w_ui_of(self);
        __w_arrange(ui, self, x, y, w, h);
        ui.layoutW = x + w;
        ui.layoutH = y + h;
        ui.needsLayout = 0;
        return;
    }
    ?&GuiUi u = self.ui();
    match (u) {
        case some(ui): { __w_arrange(ui, self, x, y, w, h); }
        case null: {
            &GuiUi tmp = GuiUi::new(self);
            __w_arrange(tmp, self, x, y, w, h);
        }
    }
}

void Widget::render(&GuiWindow win) {
    &GuiUi ui = __w_ui_of(self);
    // A tree laid out by hand to a box other than the window keeps it.
    int manual = (ui.layoutW >= 0 && !ui.needsLayout && (self.x != 0 || self.y != 0 ||
                  self.w != win.width || self.h != win.height)) ? 1 : 0;
    ui.win = win;
    float s = (float)win.scale;
    if (s <= 0.0f) { s = 1.0f; }
    if (s != ui.scale) { ui.scale = s; ui.needsLayout = 1; }
    if (manual) { ui.layoutW = win.width; ui.layoutH = win.height; }
    ui.render();
}

int Widget::dispatch_event(&GuiEvent ev) {
    &GuiUi ui = __w_ui_of(self);
    if (ui.layoutW < 0) {
        // never laid out: use the current rects as they are
        ui.layoutW = self.x + self.w;
        ui.layoutH = self.y + self.h;
        ui.needsLayout = 0;
    }
    int keep = ui.needsLayout;
    if (ui.win.is_null()) { ui.needsLayout = 0; }   // no window to size from: keep the rects
    int r = ui.dispatch(ev);
    if (ui.win.is_null()) { ui.needsLayout = keep | ui.needsLayout; }
    return r;
}

void Widget::free() {
    __w_ui_forget(self);
    unsigned long i = 0UL;
    while (i < self.children.len) {
        &Widget child = *self.children.at(i);
        child.parent = null;
        child.free();
        i = i + 1UL;
    }
    self.children.free();
    self.text.free();
    self.name.free();
    self.tooltip.free();
    self.placeholder.free();
    self.shortcut.free();
    self.colStretch.free();
    self.rowStretch.free();
    i = 0UL;
    while (i < self.items.len) { self.items.at(i).free(); i = i + 1UL; }
    self.items.free();
    i = 0UL;
    while (i < self.nodes.len) { self.nodes.at(i).text.free(); i = i + 1UL; }
    self.nodes.free();
    i = 0UL;
    while (i < self.tableCols.len) { self.tableCols.at(i).title.free(); i = i + 1UL; }
    self.tableCols.free();
    i = 0UL;
    while (i < self.cells.len) { self.cells.at(i).free(); i = i + 1UL; }
    self.cells.free();
    match (self.contextMenu) {
        case some(m): {
            m.owner = null;
            if (m.open) {
                ?&GuiUi u = self.ui();
                match (u) { case some(ui): { __w_popup_close_from(ui, m); } case null: { } }
            }
            m.free();
        }
        case null: { }
    }
    match (self.popup) {
        case some(p): {
            if (p.kind == WIDGET_POPUPLIST) {
                if (p.open) {
                    ?&GuiUi u = self.ui();
                    match (u) { case some(ui): { __w_popup_close_from(ui, p); } case null: { } }
                }
                p.free();
            }
        }
        case null: { }
    }
    unsafe { dealloc_raw_((void*)self); }
}

} // namespace std
