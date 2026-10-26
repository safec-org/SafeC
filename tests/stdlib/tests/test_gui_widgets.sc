// std::gui widget toolkit: layout (boxes, grids, stacks, scroll views,
// margins, min/max), keyboard focus and Tab navigation, text editing with
// the clipboard, lists/combos/trees/tables, menus and shortcuts, context
// menus, dialogs and message boxes, tooltips, tabs, splitters, values
// (slider/spinner/color picker), themes and HiDPI scaling — all driven
// by synthetic events through the headless backend.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/collections/string.h>
#include <std/gui/gui.h>
#include <std/gui/gui_headless.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_text.h>
#include <std/gui/gui_widget.h>

static int gClicks = 0;
static int gChanges = 0;
static int gActivates = 0;
static int gResult = -1;
static int gDraws = 0;

static void on_click_cb(&Widget w, ?&Erased d) { gClicks = gClicks + 1; }
static void on_change_cb(&Widget w, ?&Erased d) { gChanges = gChanges + 1; }
static void on_activate_cb(&Widget w, ?&Erased d) { gActivates = gActivates + 1; }
static void on_close_cb(&Widget w, int result, ?&Erased d) { gResult = result; }
static void on_draw_cb(&Widget w, &GuiCanvas c, ?&Erased d) {
    gDraws = gDraws + 1;
    c.set_fill_color(GuiColor::hex(0xFF0000U));
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
}

static void reset_counters() { gClicks = 0; gChanges = 0; gActivates = 0; gResult = -1; gDraws = 0; }

// A window, a UI for 'root' with the light theme on the built-in font
// (so nothing depends on the fonts installed), laid out and painted.
static &GuiUi make_ui(&GuiWindow win, &Widget root) {
    &GuiUi ui = GuiUi::new(root);
    struct GuiTheme t = GuiTheme::light();
    t.font = GuiTypeface::fallback();
    t.boldFont = null;
    ui.set_theme(t);
    ui.attach(win);
    ui.render();
    return ui;
}

// Delivers the queued events to the UI, then repaints.
static void flush(&GuiWindow win, &GuiUi ui) {
    struct GuiEvent ev;
    while (win.poll_event(&ev)) { ui.dispatch(&ev); }
    ui.render();
}

extern int strcmp(const char* a, const char* b);
static int streq(const &char a, const &char b) { unsafe { return strcmp((const char*)a, (const char*)b) == 0 ? 1 : 0; } }

static double cx(const &Widget w) { return (double)w.x + (double)w.w / 2.0; }
static double cy(const &Widget w) { return (double)w.y + (double)w.h / 2.0; }

static void click(&GuiWindow win, &GuiUi ui, double x, double y) {
    gui_headless_click(win, x, y);
    flush(win, ui);
}
static void click_w(&GuiWindow win, &GuiUi ui, const &Widget w) { click(win, ui, cx(w), cy(w)); }
static void key(&GuiWindow win, &GuiUi ui, int k, int mods) {
    gui_headless_key(win, k, mods, 0U);
    flush(win, ui);
}
static void type(&GuiWindow win, &GuiUi ui, const &char s) {
    gui_headless_type(win, s);
    flush(win, ui);
}
static int is_focus(const &GuiUi ui, const &Widget w) {
    match (ui.focus) {
        case some(f): { unsafe { return ((const unsigned char*)f == (const unsigned char*)w) ? 1 : 0; } }
        case null: { }
    }
    return 0;
}

// ── layout ──────────────────────────────────────────────────────────────

void t_box_layout() {
    struct GuiWindow win = GuiWindow::create("layout", 400, 300);
    &Widget root = Widget::vstack().set_padding(10).set_spacing(5);
    &Widget a = Widget::panel().set_pref_size(100, 30).set_alignment(WIDGET_ALIGN_START);
    &Widget row = Widget::hstack().set_spacing(0).set_pref_size(0, 20);
    &Widget b = Widget::panel().set_stretch(1).set_padding(0);
    &Widget c = Widget::panel().set_stretch(2).set_padding(0);
    &Widget d = Widget::panel().set_pref_size(40, 20);
    row.add_child(b); row.add_child(c); row.add_child(d);
    &Widget centered = Widget::panel().set_pref_size(60, 10).set_alignment(WIDGET_ALIGN_CENTER);
    &Widget ended = Widget::panel().set_pref_size(60, 10).set_alignment(WIDGET_ALIGN_END);
    &Widget capped = Widget::panel().set_stretch(1).set_max_size(0, 50);
    &Widget rest = Widget::panel().set_stretch(1).set_margin(4);
    root.add_child(a); root.add_child(row); root.add_child(centered); root.add_child(ended);
    root.add_child(capped); root.add_child(rest);
    &GuiUi ui = make_ui(&win, root);

    ASSERT_EQ(root.x, 0); ASSERT_EQ(root.w, 400); ASSERT_EQ(root.h, 300);
    ASSERT_EQ(a.x, 10); ASSERT_EQ(a.y, 10); ASSERT_EQ(a.w, 100); ASSERT_EQ(a.h, 30);
    // The row stretches across (380 px); 340 px are shared 1:2.
    ASSERT_EQ(row.x, 10); ASSERT_EQ(row.y, 45); ASSERT_EQ(row.w, 380); ASSERT_EQ(row.h, 20);
    ASSERT_EQ(b.x, 10);
    ASSERT_EQ(b.w + c.w, 340);
    ASSERT_TRUE(c.w - 2 * b.w <= 2 && 2 * b.w - c.w <= 2);
    ASSERT_EQ(c.x, b.x + b.w);
    ASSERT_EQ(d.x, 350); ASSERT_EQ(d.w, 40);
    ASSERT_EQ(centered.x, 10 + (380 - 60) / 2);
    ASSERT_EQ(ended.x, 390 - 60);
    // The two stretchy panels share what's left; 'capped' stops at 50.
    ASSERT_EQ(capped.h, 50);
    ASSERT_EQ(rest.x, 14); ASSERT_EQ(rest.w, 372);
    ASSERT_EQ(rest.y + rest.h, 300 - 10 - 4);
    ASSERT_EQ(rest.y, capped.y + capped.h + 5 + 4);

    // Resizing the window lays the tree out again.
    gui_headless_resize(&win, 500, 200);
    flush(&win, ui);
    ASSERT_EQ(root.w, 500); ASSERT_EQ(row.w, 480);
    ASSERT_EQ(d.x, 450);
    ASSERT_EQ(rest.y + rest.h, 200 - 14);

    // Hidden widgets take no space.
    b.set_visible(0);
    flush(&win, ui);
    ASSERT_EQ(c.x, 10); ASSERT_EQ(c.w, 440);
    ui.free();
    win.destroy();
}

void t_grid_stack_layout() {
    struct GuiWindow win = GuiWindow::create("grid", 300, 200);
    &Widget root = Widget::vstack();
    &Widget g = Widget::grid(3).set_spacing(0).set_alignment(WIDGET_ALIGN_STRETCH);
    &Widget cells[6];
    int i = 0;
    while (i < 5) {
        cells[i] = Widget::panel().set_pref_size(50, 20).set_align2(WIDGET_ALIGN_STRETCH, WIDGET_ALIGN_STRETCH);
        g.add_child(cells[i]);
        i = i + 1;
    }
    // Spans two columns: auto placement continues after it.
    cells[1].set_span(1, 2);
    g.set_column_stretch(2, 1);
    root.add_child(g);
    &Widget st = Widget::stack().set_stretch(1);
    &Widget under = Widget::panel();
    &Widget over = Widget::label("on top").set_align2(WIDGET_ALIGN_CENTER, WIDGET_ALIGN_CENTER);
    st.add_child(under); st.add_child(over);
    root.add_child(st);
    &GuiUi ui = make_ui(&win, root);

    // Row 0: cells 0 and 1 (spanning columns 1-2); row 1: cells 2, 3, 4.
    ASSERT_EQ(cells[0].x, 0); ASSERT_EQ(cells[0].y, 0); ASSERT_EQ(cells[0].w, 50);
    ASSERT_EQ(cells[1].x, 50); ASSERT_EQ(cells[1].y, 0);
    ASSERT_EQ(cells[1].w, 250);                       // column 2 stretches to the edge
    ASSERT_EQ(cells[2].x, 0); ASSERT_EQ(cells[2].y, 20);
    ASSERT_EQ(cells[3].x, 50); ASSERT_EQ(cells[4].x, 100);
    ASSERT_EQ(cells[4].w, 200);
    ASSERT_EQ(g.h, 40);
    // The stack fills the rest; its children overlap.
    ASSERT_EQ(st.y, 40 + ui.theme.spacing);
    ASSERT_EQ(under.x, st.x); ASSERT_EQ(under.y, st.y); ASSERT_EQ(under.w, st.w); ASSERT_EQ(under.h, st.h);
    ASSERT_TRUE(over.w < st.w);
    ASSERT_TRUE(over.x + over.w / 2 - (st.x + st.w / 2) <= 1 && (st.x + st.w / 2) - (over.x + over.w / 2) <= 1);

    // Explicit placement.
    cells[4].set_cell(2, 0).set_span(1, 3);
    flush(&win, ui);
    ASSERT_EQ(cells[4].x, 0); ASSERT_EQ(cells[4].y, 40); ASSERT_EQ(cells[4].w, 300);
    ui.free();
    win.destroy();
}

void t_scroll_view() {
    struct GuiWindow win = GuiWindow::create("scroll", 200, 200);
    &Widget root = Widget::vstack();
    &Widget sc = Widget::scroll().set_stretch(1);
    &Widget content = Widget::vstack().set_spacing(0);
    int i = 0;
    while (i < 20) {
        content.add_child(Widget::panel().set_pref_size(0, 30));
        i = i + 1;
    }
    sc.add_child(content);
    root.add_child(sc);
    &GuiUi ui = make_ui(&win, root);
    ASSERT_TRUE(sc.contentH >= 600.0f);
    ASSERT_TRUE(sc.scrollY == 0.0f);
    int y0 = content.y;
    gui_headless_scroll(&win, 100.0, 100.0, 0.0, -2.0);     // two lines down
    flush(&win, ui);
    ASSERT_TRUE(sc.scrollY > 0.0f);
    ASSERT_EQ(content.y, y0 - (int)sc.scrollY);
    // Scrolling far clamps to the end.
    gui_headless_scroll(&win, 100.0, 100.0, 0.0, -1000.0);
    flush(&win, ui);
    ASSERT_TRUE(sc.scrollY <= sc.contentH - (float)sc.h + 0.5f);
    ASSERT_TRUE(sc.scrollY >= sc.contentH - (float)sc.h - (float)ui.theme.scrollbarSize - 1.0f);
    gui_headless_scroll(&win, 100.0, 100.0, 0.0, 1000.0);
    flush(&win, ui);
    ASSERT_TRUE(sc.scrollY == 0.0f);
    ui.free();
    win.destroy();
}

// ── focus and keyboard ──────────────────────────────────────────────────

void t_focus_and_keys() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("focus", 300, 300);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget input = Widget::textinput("name");
    &Widget btn = Widget::button("Go").on_click(on_click_cb, (?&Erased)null);
    &Widget chk = Widget::checkbox("Check", 0).on_change(on_change_cb, (?&Erased)null);
    &Widget off = Widget::button("Off").set_enabled(0);
    &Widget sl = Widget::slider(0.0, 10.0, 5.0).set_step(1.0);
    &Widget lbl = Widget::label("not focusable");
    root.add_child(input); root.add_child(btn); root.add_child(chk);
    root.add_child(off); root.add_child(sl); root.add_child(lbl);
    &GuiUi ui = make_ui(&win, root);

    ASSERT_TRUE(ui.focus.is_null());
    key(&win, ui, GUI_KEY_TAB, 0);
    ASSERT_TRUE(is_focus(ui, input));
    ASSERT_EQ(input.focused, 1);
    key(&win, ui, GUI_KEY_TAB, 0);
    ASSERT_TRUE(is_focus(ui, btn));
    ASSERT_EQ(input.focused, 0);
    key(&win, ui, GUI_KEY_SPACE, 0);
    key(&win, ui, GUI_KEY_ENTER, 0);
    ASSERT_EQ(gClicks, 2);
    key(&win, ui, GUI_KEY_TAB, 0);
    ASSERT_TRUE(is_focus(ui, chk));
    key(&win, ui, GUI_KEY_SPACE, 0);
    ASSERT_EQ(chk.is_checked(), 1);
    ASSERT_EQ(gChanges, 1);
    key(&win, ui, GUI_KEY_TAB, 0);                    // skips the disabled button
    ASSERT_TRUE(is_focus(ui, sl));
    key(&win, ui, GUI_KEY_RIGHT, 0);
    ASSERT_TRUE(sl.value() == 6.0);
    key(&win, ui, GUI_KEY_END, 0);
    ASSERT_TRUE(sl.value() == 10.0);
    key(&win, ui, GUI_KEY_TAB, 0);                    // wraps around
    ASSERT_TRUE(is_focus(ui, input));
    key(&win, ui, GUI_KEY_TAB, GUI_MOD_SHIFT);
    ASSERT_TRUE(is_focus(ui, sl));
    key(&win, ui, GUI_KEY_TAB, GUI_MOD_SHIFT);
    ASSERT_TRUE(is_focus(ui, chk));
    // Escape drops the focus; a click focuses what it hits.
    key(&win, ui, GUI_KEY_ESCAPE, 0);
    ASSERT_TRUE(ui.focus.is_null());
    click_w(&win, ui, btn);
    ASSERT_EQ(gClicks, 3);
    ASSERT_TRUE(is_focus(ui, btn));
    // Clicking a disabled button does nothing; clicking a label keeps the focus.
    click_w(&win, ui, off);
    ASSERT_EQ(gClicks, 3);
    click_w(&win, ui, lbl);
    ASSERT_TRUE(is_focus(ui, btn));
    // Hover state and the cursor follow the pointer.
    gui_headless_mouse_move(&win, cx(input), cy(input));
    flush(&win, ui);
    ASSERT_EQ(input.hover, 1);
    ASSERT_EQ(gui_headless_cursor(&win), GUI_CURSOR_IBEAM);
    gui_headless_mouse_move(&win, cx(btn), cy(btn));
    flush(&win, ui);
    ASSERT_EQ(input.hover, 0);
    ASSERT_EQ(btn.hover, 1);
    ASSERT_EQ(gui_headless_cursor(&win), GUI_CURSOR_ARROW);
    ui.free();
    win.destroy();
}

// ── text ────────────────────────────────────────────────────────────────

void t_text_editing() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("text", 400, 300);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget input = Widget::textinput("type").on_change(on_change_cb, (?&Erased)null)
                                             .on_activate(on_activate_cb, (?&Erased)null);
    &Widget area = Widget::textarea().set_pref_size(0, 100);
    &Widget ro = Widget::textinput("").set_text("fixed").set_read_only(1);
    &Widget pw = Widget::textinput("").set_password(1);
    root.add_child(input); root.add_child(area); root.add_child(ro); root.add_child(pw);
    &GuiUi ui = make_ui(&win, root);
    int pm = gui_primary_mod();

    click_w(&win, ui, input);
    ASSERT_TRUE(is_focus(ui, input));
    type(&win, ui, "hello world");
    ASSERT_TRUE(input.text.eq_cstr("hello world"));
    ASSERT_EQ(input.caret, 11UL);
    ASSERT_EQ(gChanges, 11);
    key(&win, ui, GUI_KEY_HOME, 0);
    ASSERT_EQ(input.caret, 0UL);
    key(&win, ui, GUI_KEY_RIGHT, pm == GUI_MOD_META ? GUI_MOD_ALT : GUI_MOD_CTRL);   // a word right
    ASSERT_EQ(input.caret, 5UL);
    key(&win, ui, GUI_KEY_END, GUI_MOD_SHIFT);
    ASSERT_EQ(input.selection_start(), 5UL);
    ASSERT_EQ(input.selection_end(), 11UL);
    gui_headless_key(&win, (int)'C', pm, 0U);
    flush(&win, ui);
    struct String clip = String::new();
    ASSERT_TRUE(gui_clipboard_get_text(&clip));
    ASSERT_TRUE(clip.eq_cstr(" world"));
    // Typing replaces the selection.
    type(&win, ui, "!");
    ASSERT_TRUE(input.text.eq_cstr("hello!"));
    gui_headless_key(&win, (int)'V', pm, 0U);
    flush(&win, ui);
    ASSERT_TRUE(input.text.eq_cstr("hello! world"));
    // Backspace a word; Delete a character; select all and cut.
    key(&win, ui, GUI_KEY_BACKSPACE, pm == GUI_MOD_META ? GUI_MOD_ALT : GUI_MOD_CTRL);
    ASSERT_TRUE(input.text.eq_cstr("hello! "));
    key(&win, ui, GUI_KEY_HOME, 0);
    key(&win, ui, GUI_KEY_DELETE, 0);
    ASSERT_TRUE(input.text.eq_cstr("ello! "));
    gui_headless_key(&win, (int)'A', pm, 0U);
    gui_headless_key(&win, (int)'X', pm, 0U);
    flush(&win, ui);
    ASSERT_EQ(input.text.len, 0UL);
    clip.clear();
    ASSERT_TRUE(gui_clipboard_get_text(&clip));
    ASSERT_TRUE(clip.eq_cstr("ello! "));
    // UTF-8: one key per character, the caret moves by characters.
    type(&win, ui, "h\xC3\xA9llo");                   // héllo
    ASSERT_EQ(input.text.len, 6UL);
    key(&win, ui, GUI_KEY_HOME, 0);
    key(&win, ui, GUI_KEY_RIGHT, 0);
    key(&win, ui, GUI_KEY_RIGHT, 0);
    ASSERT_EQ(input.caret, 3UL);
    key(&win, ui, GUI_KEY_BACKSPACE, 0);
    ASSERT_TRUE(input.text.eq_cstr("hllo"));
    // Enter activates a single-line field.
    key(&win, ui, GUI_KEY_ENTER, 0);
    ASSERT_EQ(gActivates, 1);
    // Mouse: a double click selects a word.
    input.set_text("alpha beta");
    flush(&win, ui);
    double wx = (double)input.x + 8.0 + (double)GuiTypeface::fallback().measure("alpha be", ui.theme.fontSize);
    gui_headless_click(&win, wx, cy(input));
    gui_headless_click(&win, wx, cy(input));
    flush(&win, ui);
    ASSERT_EQ(input.selection_start(), 6UL);
    ASSERT_EQ(input.selection_end(), 10UL);

    // A text area takes Enter as a new line, Up/Down move between lines.
    click_w(&win, ui, area);
    type(&win, ui, "one");
    key(&win, ui, GUI_KEY_ENTER, 0);
    type(&win, ui, "two");
    ASSERT_TRUE(area.text.eq_cstr("one\ntwo"));
    key(&win, ui, GUI_KEY_UP, 0);
    ASSERT_EQ(area.caret, 3UL);
    key(&win, ui, GUI_KEY_DOWN, 0);
    ASSERT_EQ(area.caret, 7UL);
    // Read-only: selectable, not editable.
    click_w(&win, ui, ro);
    type(&win, ui, "x");
    ASSERT_TRUE(ro.text.eq_cstr("fixed"));
    // Password fields don't copy their text out.
    click_w(&win, ui, pw);
    type(&win, ui, "secret");
    gui_headless_key(&win, (int)'A', pm, 0U);
    gui_headless_key(&win, (int)'C', pm, 0U);
    flush(&win, ui);
    clip.clear();
    gui_clipboard_get_text(&clip);
    ASSERT_FALSE(clip.eq_cstr("secret"));
    ASSERT_TRUE(pw.text.eq_cstr("secret"));
    clip.free();
    ui.free();
    win.destroy();
}

// ── item views ──────────────────────────────────────────────────────────

void t_lists_and_combos() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("lists", 400, 400);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget cb = Widget::combo().on_change(on_change_cb, (?&Erased)null);
    cb.add_item("Red"); cb.add_item("Green"); cb.add_item("Blue");
    &Widget lst = Widget::list().set_pref_size(0, 120);
    lst.add_item("One"); lst.add_item("Two"); lst.add_item("Three");
    &Widget rg = Widget::radio_group();
    rg.add_item("A"); rg.add_item("B");
    root.add_child(cb); root.add_child(lst); root.add_child(rg);
    &GuiUi ui = make_ui(&win, root);

    ASSERT_EQ(cb.selected_index(), 0);               // a combo box starts on its first item
    ASSERT_EQ(lst.item_count(), 3UL);
    ASSERT_TRUE(streq(lst.item_text(1UL), "Two"));
    ASSERT_TRUE(streq(lst.item_text(9UL), ""));
    // Opening the combo shows its list as a popup; clicking an entry picks it.
    click_w(&win, ui, cb);
    ASSERT_EQ(ui.overlays.len, 1UL);
    &Widget pop = *ui.overlays.at(0UL);
    ASSERT_TRUE(pop.y >= cb.y + cb.h - 1);
    double rowH = (double)pop.h / 3.0;
    click(&win, ui, cx(pop), (double)pop.y + rowH * 1.5);
    ASSERT_EQ(ui.overlays.len, 0UL);
    ASSERT_EQ(cb.selected_index(), 1);
    ASSERT_EQ(gChanges, 1);
    // Keyboard: Down picks the next entry without opening.
    key(&win, ui, GUI_KEY_DOWN, 0);
    ASSERT_EQ(cb.selected_index(), 2);
    ASSERT_EQ(gChanges, 2);
    // A click outside closes an open popup without doing anything else.
    click_w(&win, ui, cb);
    ASSERT_EQ(ui.overlays.len, 1UL);
    click(&win, ui, 390.0, 390.0);
    ASSERT_EQ(ui.overlays.len, 0UL);
    ASSERT_EQ(cb.selected_index(), 2);

    // Lists: click a row, arrow keys, Home/End.
    click(&win, ui, cx(lst), (double)lst.y + 4.0 + (double)ui.theme.rowHeight * 1.5);
    ASSERT_EQ(lst.selected_index(), 1);
    key(&win, ui, GUI_KEY_DOWN, 0);
    ASSERT_EQ(lst.selected_index(), 2);
    key(&win, ui, GUI_KEY_DOWN, 0);
    ASSERT_EQ(lst.selected_index(), 2);
    key(&win, ui, GUI_KEY_HOME, 0);
    ASSERT_EQ(lst.selected_index(), 0);
    // Radio groups: click the second option.
    click(&win, ui, (double)rg.x + 10.0, (double)rg.y + (double)rg.h * 0.75);
    ASSERT_EQ(rg.selected_index(), 1);
    key(&win, ui, GUI_KEY_UP, 0);
    ASSERT_EQ(rg.selected_index(), 0);
    ui.free();
    win.destroy();
}

void t_tree_and_table() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("tree", 400, 400);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget tr = Widget::tree().set_pref_size(0, 150).on_change(on_change_cb, (?&Erased)null);
    int top = tr.add_node(-1, "Root");
    int kid = tr.add_node(top, "Child");
    tr.add_node(kid, "Grandchild");
    int other = tr.add_node(-1, "Other");
    &Widget tb = Widget::table().set_stretch(1);
    tb.add_column("Name", 120); tb.add_column("Size", 0);
    int r0 = tb.add_row(); tb.set_cell_text(r0, 0, "a.txt"); tb.set_cell_text(r0, 1, "1 KB");
    int r1 = tb.add_row(); tb.set_cell_text(r1, 0, "b.txt"); tb.set_cell_text(r1, 1, "2 KB");
    root.add_child(tr); root.add_child(tb);
    &GuiUi ui = make_ui(&win, root);

    ASSERT_EQ(tr.node_count(), 4);
    ASSERT_TRUE(streq(tr.node_text(kid), "Child"));
    // Collapsed: rows are Root, Other. Clicking the second row selects "Other".
    double rh = (double)ui.theme.rowHeight;
    click(&win, ui, cx(tr), (double)tr.y + 4.0 + rh * 1.5);
    ASSERT_EQ(tr.selected_index(), other);
    // Expand Root from the keyboard: Up, Right; then Down reaches Child.
    key(&win, ui, GUI_KEY_UP, 0);
    ASSERT_EQ(tr.selected_index(), top);
    key(&win, ui, GUI_KEY_RIGHT, 0);
    ASSERT_EQ(tr.nodes.at((unsigned long)top).expanded, 1);
    key(&win, ui, GUI_KEY_DOWN, 0);
    ASSERT_EQ(tr.selected_index(), kid);
    key(&win, ui, GUI_KEY_LEFT, 0);                   // to the parent
    ASSERT_EQ(tr.selected_index(), top);
    key(&win, ui, GUI_KEY_LEFT, 0);                   // collapse
    ASSERT_EQ(tr.nodes.at((unsigned long)top).expanded, 0);
    ASSERT_TRUE(gChanges >= 3);

    ASSERT_EQ(tb.row_count(), 2);
    ASSERT_TRUE(streq(tb.cell_text(r1, 1), "2 KB"));
    ASSERT_TRUE(streq(tb.cell_text(5, 5), ""));
    // Header, then rows.
    click(&win, ui, (double)tb.x + 20.0, (double)tb.y + 1.0 + rh * 2.5);
    ASSERT_EQ(tb.selected_index(), 1);
    key(&win, ui, GUI_KEY_UP, 0);
    ASSERT_EQ(tb.selected_index(), 0);
    // Dragging a column edge in the header resizes the column.
    int edge = tb.x + 1 + 120;
    gui_headless_mouse_move(&win, (double)edge, (double)tb.y + rh / 2.0);
    gui_headless_mouse_down(&win, (double)edge, (double)tb.y + rh / 2.0, GUI_BUTTON_LEFT);
    gui_headless_mouse_move(&win, (double)edge + 30.0, (double)tb.y + rh / 2.0);
    gui_headless_mouse_up(&win, (double)edge + 30.0, (double)tb.y + rh / 2.0, GUI_BUTTON_LEFT);
    flush(&win, ui);
    ASSERT_EQ(tb.tableCols.at(0UL).width, 150);
    tb.clear_rows();
    ASSERT_EQ(tb.row_count(), 0);
    ui.free();
    win.destroy();
}

// ── menus ───────────────────────────────────────────────────────────────

void t_menus_and_shortcuts() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("menus", 400, 300);
    &Widget root = Widget::vstack();
    &Widget mb = Widget::menubar();
    &Widget file = Widget::menu("File");
    &Widget open = Widget::menu_item("Open").set_shortcut("Ctrl+O").on_click(on_click_cb, (?&Erased)null);
    &Widget wrap = Widget::menu_item("Word wrap").set_checked(0).on_change(on_change_cb, (?&Erased)null);
    &Widget sub = Widget::menu("Recent");
    &Widget recent = Widget::menu_item("a.txt").on_click(on_click_cb, (?&Erased)null);
    sub.add_child(recent);
    file.add_child(open); file.add_child(wrap); file.add_child(Widget::menu_separator()); file.add_child(sub);
    mb.add_child(file);
    root.add_child(mb);
    &Widget body = Widget::list().set_stretch(1);
    &Widget ctx = Widget::menu("");
    &Widget del = Widget::menu_item("Delete").set_shortcut("Delete").on_click(on_click_cb, (?&Erased)null);
    ctx.add_child(del);
    body.set_context_menu(ctx);
    root.add_child(body);
    &GuiUi ui = make_ui(&win, root);
    double rh = (double)ui.theme.rowHeight;

    // Shortcuts work without opening the menu.
    gui_headless_key(&win, (int)'O', gui_primary_mod(), 0U);
    flush(&win, ui);
    ASSERT_EQ(gClicks, 1);
    gui_headless_key(&win, (int)'O', gui_primary_mod() | GUI_MOD_SHIFT, 0U);
    flush(&win, ui);
    ASSERT_EQ(gClicks, 1);                            // not with extra modifiers
    // A context menu's shortcuts work while its widget has the focus.
    key(&win, ui, GUI_KEY_DELETE, 0);
    ASSERT_EQ(gClicks, 1);
    click_w(&win, ui, body);
    key(&win, ui, GUI_KEY_DELETE, 0);
    ASSERT_EQ(gClicks, 2);
    // Click "File": the menu opens below the bar.
    click(&win, ui, (double)mb.x + 20.0, cy(mb));
    ASSERT_EQ(ui.overlays.len, 1UL);
    ASSERT_EQ(file.open, 1);
    ASSERT_TRUE(file.y >= mb.y + mb.h - 1);
    // Click the checkable item: it toggles and the menu closes.
    click(&win, ui, cx(wrap), cy(wrap));
    ASSERT_EQ(wrap.is_checked(), 1);
    ASSERT_EQ(gChanges, 1);
    ASSERT_EQ(ui.overlays.len, 0UL);
    // Keyboard: open, Down to "Recent", Right opens the submenu, Enter.
    click(&win, ui, (double)mb.x + 20.0, cy(mb));
    key(&win, ui, GUI_KEY_DOWN, 0);                   // Open
    key(&win, ui, GUI_KEY_DOWN, 0);                   // Word wrap
    key(&win, ui, GUI_KEY_DOWN, 0);                   // Recent (skips the separator)
    key(&win, ui, GUI_KEY_RIGHT, 0);
    ASSERT_EQ(ui.overlays.len, 2UL);
    ASSERT_TRUE(sub.x >= file.x + file.w - 4);
    key(&win, ui, GUI_KEY_ENTER, 0);
    ASSERT_EQ(gClicks, 3);
    ASSERT_EQ(ui.overlays.len, 0UL);
    // Escape closes an open menu.
    click(&win, ui, (double)mb.x + 20.0, cy(mb));
    key(&win, ui, GUI_KEY_ESCAPE, 0);
    ASSERT_EQ(ui.overlays.len, 0UL);
    // Right click: the context menu opens at the pointer.
    gui_headless_mouse_down(&win, 200.0, 150.0, GUI_BUTTON_RIGHT);
    gui_headless_mouse_up(&win, 200.0, 150.0, GUI_BUTTON_RIGHT);
    flush(&win, ui);
    ASSERT_EQ(ui.overlays.len, 1UL);
    ASSERT_TRUE(ctx.x >= 199 && ctx.x <= 201 && ctx.y >= 149 && ctx.y <= 151);
    click(&win, ui, cx(del), cy(del));
    ASSERT_EQ(gClicks, 4);
    // Near the edge, the menu is kept inside the window.
    gui_headless_mouse_down(&win, 395.0, 295.0, GUI_BUTTON_RIGHT);
    gui_headless_mouse_up(&win, 395.0, 295.0, GUI_BUTTON_RIGHT);
    flush(&win, ui);
    ASSERT_TRUE(ctx.x + ctx.w <= 400 && ctx.y + ctx.h <= 300);
    ASSERT_TRUE(rh > 0.0);
    ui.close_popups();
    ui.free();
    win.destroy();
}

// ── dialogs and tooltips ────────────────────────────────────────────────

void t_dialogs() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("dialogs", 500, 400);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget under = Widget::button("Under").on_click(on_click_cb, (?&Erased)null);
    root.add_child(under);
    &GuiUi ui = make_ui(&win, root);

    gui_message_box(ui, "Question", "Save the file?", GUI_MB_YES | GUI_MB_NO, on_close_cb, (?&Erased)null);
    flush(&win, ui);
    ASSERT_FALSE(ui.modal().is_null());
    &Widget box;
    unsafe { box = (struct Widget*)ui.modal(); }
    ASSERT_TRUE(box.x > 0 && box.x + box.w < 500);
    // The window under the dialog doesn't get clicks.
    click_w(&win, ui, under);
    ASSERT_EQ(gClicks, 0);
    ASSERT_FALSE(ui.modal().is_null());
    // Tab stays inside the dialog; the "No" button closes it.
    ?&Widget no = box.find("no");
    ASSERT_FALSE(no.is_null());
    &Widget nb;
    unsafe { nb = (struct Widget*)no; }
    click_w(&win, ui, nb);
    ASSERT_EQ(gResult, GUI_RESULT_NO);
    ASSERT_TRUE(ui.modal().is_null());
    click_w(&win, ui, under);
    ASSERT_EQ(gClicks, 1);
    // Escape cancels.
    gui_message_box(ui, "Info", "Done.", GUI_MB_OK, on_close_cb, (?&Erased)null);
    flush(&win, ui);
    key(&win, ui, GUI_KEY_ESCAPE, 0);
    ASSERT_EQ(gResult, GUI_RESULT_CANCEL);
    ASSERT_TRUE(ui.modal().is_null());
    // Enter presses the default (first) button.
    gui_message_box(ui, "Info", "Done.", GUI_MB_OK | GUI_MB_CANCEL, on_close_cb, (?&Erased)null);
    flush(&win, ui);
    key(&win, ui, GUI_KEY_ENTER, 0);
    ASSERT_EQ(gResult, GUI_RESULT_OK);
    // A custom dialog.
    &Widget dlg = Widget::dialog("Settings");
    &Widget ok = Widget::button("Close");
    dlg.add_child(Widget::checkbox("Option", 0));
    dlg.add_child(ok);
    ui.show_modal(dlg);
    flush(&win, ui);
    ASSERT_TRUE(dlg.w > 0 && dlg.x + dlg.w / 2 >= 240 && dlg.x + dlg.w / 2 <= 260);
    dlg.close_dialog(GUI_RESULT_OK);
    ASSERT_TRUE(ui.modal().is_null());
    ASSERT_EQ(dlg.result, GUI_RESULT_OK);
    dlg.free();
    ui.free();
    win.destroy();
}

void t_tooltips() {
    struct GuiWindow win = GuiWindow::create("tips", 300, 200);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget b = Widget::button("Hover me").set_tooltip("A helpful tip");
    root.add_child(b);
    &GuiUi ui = make_ui(&win, root);
    gui_headless_mouse_move(&win, cx(b), cy(b));
    flush(&win, ui);
    ui.tick(gui_time_ms());
    ASSERT_EQ(ui.tipShown, 0);
    gui_headless_advance(700.0);
    ui.tick(gui_time_ms());
    ASSERT_EQ(ui.tipShown, 1);
    ASSERT_TRUE(ui.needs_redraw());
    ui.render();
    ASSERT_TRUE(ui.tip.text.eq_cstr("A helpful tip"));
    ASSERT_TRUE(ui.tip.y >= b.y + b.h);
    // Moving away or pressing hides it.
    gui_headless_mouse_move(&win, 290.0, 190.0);
    flush(&win, ui);
    ui.tick(gui_time_ms());
    ASSERT_EQ(ui.tipShown, 0);
    ui.free();
    win.destroy();
}

// ── containers with state ───────────────────────────────────────────────

void t_tabs_and_splitter() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("tabs", 400, 300);
    &Widget root = Widget::vstack();
    &Widget tabs = Widget::tabs().set_pref_size(0, 120).on_change(on_change_cb, (?&Erased)null);
    &Widget p1 = Widget::label("first page");
    &Widget p2 = Widget::label("second page");
    tabs.add_tab("One", p1);
    tabs.add_tab("Two", p2);
    &Widget sp = Widget::splitter(WIDGET_HORIZONTAL).set_stretch(1);
    &Widget left = Widget::panel();
    &Widget right = Widget::panel();
    sp.add_child(left); sp.add_child(right);
    root.add_child(tabs); root.add_child(sp);
    &GuiUi ui = make_ui(&win, root);

    ASSERT_EQ(tabs.selected_index(), 0);
    ASSERT_EQ(p1.visible, 1);
    ASSERT_TRUE(p1.w > 0);
    ASSERT_EQ(p2.w * p2.visible, 0);
    // The second tab title sits right of the first.
    double tabY = (double)tabs.y + 10.0;
    double x2 = (double)tabs.x + 8.0 + (double)GuiTypeface::fallback().measure("One", ui.theme.fontSize) + 30.0;
    click(&win, ui, x2, tabY);
    ASSERT_EQ(tabs.selected_index(), 1);
    ASSERT_EQ(gChanges, 1);
    ASSERT_TRUE(p2.visible == 1 && p2.w > 0);
    key(&win, ui, GUI_KEY_TAB, GUI_MOD_CTRL);
    ASSERT_EQ(tabs.selected_index(), 0);

    // Splitter: half and half; dragging the divider moves it.
    ASSERT_TRUE(sp.value() == 0.5);
    int lw = left.w;
    ASSERT_TRUE(lw > 180 && lw < 200);
    ASSERT_TRUE(right.x > left.x + left.w);
    double dx = (double)(left.x + left.w) + (double)(right.x - left.x - left.w) / 2.0;
    double dy = cy(sp);
    gui_headless_mouse_move(&win, dx, dy);
    flush(&win, ui);
    ASSERT_EQ(gui_headless_cursor(&win), GUI_CURSOR_RESIZE_EW);
    gui_headless_mouse_down(&win, dx, dy, GUI_BUTTON_LEFT);
    gui_headless_mouse_move(&win, dx + 100.0, dy);
    gui_headless_mouse_up(&win, dx + 100.0, dy, GUI_BUTTON_LEFT);
    flush(&win, ui);
    ASSERT_EQ(left.w, lw + 100);
    ASSERT_TRUE(sp.value() > 0.7);
    ui.free();
    win.destroy();
}

void t_values() {
    reset_counters();
    struct GuiWindow win = GuiWindow::create("values", 400, 400);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget sl = Widget::slider(0.0, 100.0, 0.0).on_change(on_change_cb, (?&Erased)null);
    &Widget sp = Widget::spinner(0.0, 10.0, 2.0, 0.5).set_decimals(1);
    &Widget pr = Widget::progress(0.25);
    &Widget cp = Widget::color_picker(GuiColor::hex(0xFF0000U));
    &Widget cv = Widget::canvas(on_draw_cb, (?&Erased)null).set_pref_size(40, 40);
    root.add_child(sl); root.add_child(sp); root.add_child(pr); root.add_child(cp); root.add_child(cv);
    &GuiUi ui = make_ui(&win, root);

    // Clicking the slider's track jumps there and drags.
    gui_headless_mouse_down(&win, (double)sl.x + (double)sl.w * 0.5, cy(sl), GUI_BUTTON_LEFT);
    flush(&win, ui);
    ASSERT_TRUE(sl.value() > 40.0 && sl.value() < 60.0);
    gui_headless_mouse_move(&win, (double)(sl.x + sl.w + 50), cy(sl));
    gui_headless_mouse_up(&win, (double)(sl.x + sl.w + 50), cy(sl), GUI_BUTTON_LEFT);
    flush(&win, ui);
    ASSERT_TRUE(sl.value() == 100.0);
    ASSERT_TRUE(gChanges >= 2);
    // set_value clamps and doesn't notify.
    int before = gChanges;
    sl.set_value(150.0);
    ASSERT_TRUE(sl.value() == 100.0);
    ASSERT_EQ(gChanges, before);
    // Spinner: arrow keys step, typing commits on Enter / blur.
    click(&win, ui, (double)sp.x + 10.0, cy(sp));
    key(&win, ui, GUI_KEY_UP, 0);
    ASSERT_TRUE(sp.value() == 2.5);
    ASSERT_TRUE(sp.text.eq_cstr("2.5"));
    gui_headless_key(&win, (int)'A', gui_primary_mod(), 0U);
    flush(&win, ui);
    type(&win, ui, "7.25");
    key(&win, ui, GUI_KEY_ENTER, 0);
    ASSERT_TRUE(sp.value() == 7.0 || sp.value() == 7.5);
    gui_headless_key(&win, (int)'A', gui_primary_mod(), 0U);
    flush(&win, ui);
    type(&win, ui, "99");
    key(&win, ui, GUI_KEY_TAB, 0);
    ASSERT_TRUE(sp.value() == 10.0);
    // Progress values are clamped fractions.
    pr.set_value(2.0);
    ASSERT_TRUE(pr.value() == 1.0);
    // Color picker: round-trips colors, hue/sat/val follow.
    struct GuiColor c = cp.get_color();
    ASSERT_EQ((int)c.r, 255); ASSERT_EQ((int)c.g, 0); ASSERT_EQ((int)c.b, 0);
    cp.set_color(GuiColor::hex(0x3366CCU));
    c = cp.get_color();
    ASSERT_EQ((int)c.r, 0x33); ASSERT_EQ((int)c.g, 0x66); ASSERT_EQ((int)c.b, 0xCC);
    // Clicking the top-left of the saturation/value square picks white.
    click(&win, ui, (double)cp.x + 2.0, (double)cp.y + 2.0);
    c = cp.get_color();
    ASSERT_TRUE((int)c.r >= 245 && (int)c.g >= 245 && (int)c.b >= 245);
    // Canvas widgets draw in their own coordinates.
    ui.invalidate();
    ui.render();
    ASSERT_TRUE(gDraws >= 1);
    unsigned int px = win.get_pixel(cv.x + 2, cv.y + 2);
    ASSERT_EQ(px, 0xFF0000FFU);                       // RGBA, red first
    ui.free();
    win.destroy();
}

// ── themes and scaling ──────────────────────────────────────────────────

void t_themes_and_scale() {
    struct GuiWindow win = GuiWindow::create("theme", 200, 100);
    &Widget root = Widget::vstack().set_padding(8);
    &Widget b = Widget::button("Themed").set_pref_size(120, 40);
    root.add_child(b);
    &GuiUi ui = make_ui(&win, root);
    struct GuiTheme light = GuiTheme::light();
    struct GuiTheme dark = GuiTheme::dark();
    ASSERT_EQ(light.dark, 0); ASSERT_EQ(dark.dark, 1);
    ASSERT_TRUE(light.window.packed() != dark.window.packed());
    // The window background and the button face take the theme's colors.
    unsigned int bg = win.get_pixel(2, 2);
    ASSERT_EQ(bg, light.window.packed());
    unsigned int face = win.get_pixel(b.x + 6, b.y + b.h - 6);
    ASSERT_EQ(face, light.button.packed());
    dark.font = GuiTypeface::fallback();
    ui.set_theme(dark);
    ui.render();
    bg = win.get_pixel(2, 2);
    ASSERT_EQ(bg, dark.window.packed());
    ui.free();
    win.destroy();

    // A 2x window: the same logical layout in twice the pixels.
    gui_headless_set_scale(2.0);
    struct GuiWindowOptions o = GuiWindowOptions::new("hidpi", 200, 100);
    struct GuiWindow hw = GuiWindow::create_with(&o);
    ASSERT_EQ(hw.width, 400); ASSERT_EQ(hw.height, 200);
    &Widget r2 = Widget::vstack().set_padding(8);
    &Widget b2 = Widget::button("Themed").set_pref_size(120, 40);
    r2.add_child(b2);
    &GuiUi ui2 = make_ui(&hw, r2);
    ASSERT_EQ(b2.x, 16); ASSERT_EQ(b2.w, 240); ASSERT_EQ(b2.h, 80);
    // Clicks arrive in buffer pixels too.
    reset_counters();
    b2.on_click(on_click_cb, (?&Erased)null);
    click_w(&hw, ui2, b2);
    ASSERT_EQ(gClicks, 1);
    ui2.free();
    hw.destroy();
    gui_headless_set_scale(1.0);
}

// ── tree operations ─────────────────────────────────────────────────────

void t_tree_ops() {
    &Widget root = Widget::vstack();
    &Widget a = Widget::label("a").set_name("a");
    &Widget box = Widget::hstack();
    &Widget b = Widget::button("b").set_name("b");
    box.add_child(b);
    root.add_child(a);
    root.add_child(box);
    ASSERT_EQ(root.child_count(), 2UL);
    ?&Widget f = root.find("b");
    ASSERT_FALSE(f.is_null());
    ASSERT_TRUE(root.find("zzz").is_null());
    root.remove_child(a);
    ASSERT_EQ(root.child_count(), 1UL);
    ASSERT_TRUE(a.parent.is_null());
    root.insert_child(0UL, a);
    match (root.child_at(0UL)) {
        case some(w): { ASSERT_TRUE(w.text.eq_cstr("a")); }
        case null: { ASSERT_TRUE(0); }
    }
    box.clear_children();
    ASSERT_EQ(box.child_count(), 0UL);
    root.free();
}

int main() {
    gui_use_headless();
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("box layout: stretch, align, margins, max size, resize", t_box_layout);
        t.add("grid and stack layout", t_grid_stack_layout);
        t.add("scroll views", t_scroll_view);
        t.add("focus, Tab navigation, keys, hover, cursors", t_focus_and_keys);
        t.add("text editing and the clipboard", t_text_editing);
        t.add("combo boxes, lists, radio groups", t_lists_and_combos);
        t.add("trees and tables", t_tree_and_table);
        t.add("menus, shortcuts, context menus", t_menus_and_shortcuts);
        t.add("dialogs and message boxes", t_dialogs);
        t.add("tooltips", t_tooltips);
        t.add("tabs and splitters", t_tabs_and_splitter);
        t.add("sliders, spinners, progress, color picker, canvas", t_values);
        t.add("themes and HiDPI scaling", t_themes_and_scale);
        t.add("tree operations", t_tree_ops);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
