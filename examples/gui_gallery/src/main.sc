// gui_gallery — every std/gui widget, in two windows run by one GuiApp:
//
//   "Widgets"  a menu bar, tabs of controls (buttons, check boxes,
//              toggles, radio groups, sliders, progress bars, spinners,
//              combo boxes, text fields), data views (list, tree, table
//              in splitters, a context menu), pickers (color picker,
//              image view, canvas widget), layouts (grid, scroll view,
//              stack), tooltips, dialogs and a status bar.
//   "Canvas"   the vector canvas on its own: paths, gradients, strokes,
//              dashes, blend modes, clipping and text, animated.
//
// The light/dark theme follows the system ($SAFEC_GUI_THEME=dark|light
// overrides it); View > Dark theme switches both windows.
//
//   safeguard run                         the windows
//   safeguard run -- --snapshot DIR       renders both windows headless to
//                                         DIR/widgets.png and DIR/canvas.png
#include <std/gui/gui.h>
#include <std/gui/gui_headless.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_png.h>
#include <std/gui/gui_widget.h>
#include <std/gui/gui_app.h>
#include <std/collections/string.h>
#include <std/math.h>

extern int printf(const char* fmt, ...);
extern int snprintf(char* buf, unsigned long n, const char* fmt, ...);
extern int strcmp(const char* a, const char* b);

struct Gallery {
    &GuiApp app;
    ?&GuiWindow widgets;
    ?&GuiWindow canvas;
    &Widget status;
    &Widget progress;
    &Widget busy;
    &Widget slider;
    &Widget sliderLabel;
    &Widget picker;
    &Widget swatch;
    &Widget input;
    &Widget table;
    &Widget dark;
    &Widget settings;              // the settings dialog, shown on demand
    struct GuiImage photo;
    double t;
    int clicks;
};

// ── helpers ───────────────────────────────────────────────────────────────

static void set_status(&Gallery g, const &char text) { g.status.set_text(text); }

static void apply_theme(&Gallery g, int dark) {
    struct GuiTheme t = dark ? GuiTheme::dark() : GuiTheme::light();
    match (g.widgets) {
        case some(w): { match (g.app.ui_of(w)) { case some(ui): { ui.set_theme(t); } case null: { } } }
        case null: { }
    }
    match (g.canvas) { case some(w): { g.app.redraw(w); } case null: { } }
}

static int is_dark(&Gallery g) { return g.dark.is_checked(); }

// ── callbacks ─────────────────────────────────────────────────────────────

static void on_button(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            g.clicks = g.clicks + 1;
            char buf[96];
            unsafe { snprintf(&buf[0], 96UL, "\"%s\" clicked (%d clicks so far)", (const char*)w.text.data, g.clicks); }
            set_status(g, &buf[0]);
        }
        case null: { }
    }
}

static void on_slider(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            char buf[32];
            unsafe { snprintf(&buf[0], 32UL, "%.0f %%", w.value()); }
            g.sliderLabel.set_text(&buf[0]);
            g.progress.set_value(w.value() / 100.0);
        }
        case null: { }
    }
}

static void on_color(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            struct GuiColor c = w.get_color();
            char buf[64];
            unsafe { snprintf(&buf[0], 64UL, "Picked #%02X%02X%02X", (unsigned int)c.r, (unsigned int)c.g, (unsigned int)c.b); }
            set_status(g, &buf[0]);
            match (g.canvas) { case some(cw): { g.app.redraw(cw); } case null: { } }
        }
        case null: { }
    }
}

static void on_input(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            char buf[128];
            unsafe { snprintf(&buf[0], 128UL, "Typed: %s", (const char*)w.text.data); }
            set_status(g, &buf[0]);
        }
        case null: { }
    }
}

static void on_select(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            char buf[96];
            int i = w.selected_index();
            const char* what = "";
            unsafe {
                if (w.kind == WIDGET_TREE && i >= 0) { what = (const char*)w.node_text(i); }
                else if (w.kind == WIDGET_TABLE && i >= 0) { what = (const char*)w.cell_text(i, 0); }
                else if (i >= 0) { what = (const char*)w.item_text((unsigned long)i); }
                snprintf(&buf[0], 96UL, "Selected #%d %s", i, what);
            }
            set_status(g, &buf[0]);
        }
        case null: { }
    }
}

static void on_dark(&Widget w, ?&Gallery og) {
    match (og) { case some(g): { apply_theme(g, w.is_checked()); } case null: { } }
}

static void on_quit(&Widget w, ?&Gallery og) {
    match (og) { case some(g): { g.app.quit(0); } case null: { } }
}

static void on_box_closed(&Widget box, int result, ?&Gallery og) {
    match (og) {
        case some(g): {
            set_status(g, result == GUI_RESULT_YES ? "You said yes." :
                          (result == GUI_RESULT_NO ? "You said no." : "Cancelled."));
        }
        case null: { }
    }
}

static void on_message_box(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            match (w.ui()) {
                case some(ui): {
                    gui_message_box(ui, "A question", "Do you like widgets? This is a modal message box: the window behind it doesn't react until you answer.",
                                    GUI_MB_YES | GUI_MB_NO | GUI_MB_CANCEL, on_box_closed, g);
                }
                case null: { }
            }
        }
        case null: { }
    }
}

static void on_dialog_close(&Widget w, ?&Gallery og) {
    // the button's dialog is its parent's parent (dialog > row > button)
    ?&Widget p = w.parent;
    while (!p.is_null()) {
        &Widget d;
        unsafe { d = (struct Widget*)p; }
        if (d.kind == WIDGET_DIALOG) {
            d.close_dialog(GUI_RESULT_OK);
            match (og) { case some(g): { set_status(g, "Settings saved."); } case null: { } }
            return;
        }
        p = d.parent;
    }
}

static &Widget make_settings_dialog(&Gallery g) {
    &Widget dlg = Widget::dialog("Settings");
    &Widget grid = Widget::grid(2);
    grid.add_child(Widget::label("Name"));
    grid.add_child(Widget::textinput("Your name").set_min_size(200, 0));
    grid.add_child(Widget::label("Volume"));
    grid.add_child(Widget::slider(0.0, 10.0, 7.0));
    grid.add_child(Widget::label("Autosave"));
    grid.add_child(Widget::toggle("", 1));
    grid.set_column_stretch(1, 1);
    dlg.add_child(grid);
    &Widget row = Widget::hstack();
    row.add_child(Widget::spacer().set_stretch(1));
    row.add_child(Widget::button("Save").on_click(on_dialog_close, g));
    dlg.add_child(row);
    return dlg;
}

static void on_settings(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            match (w.ui()) {
                // one dialog, shown again and again (a UI doesn't own the
                // dialogs it shows: main() frees it at the end)
                case some(ui): { if (ui.modal().is_null()) { ui.show_modal(g.settings); } }
                case null: { }
            }
        }
        case null: { }
    }
}

static void on_add_row(&Widget w, ?&Gallery og) {
    match (og) {
        case some(g): {
            int r = g.table.add_row();
            char buf[32];
            unsafe { snprintf(&buf[0], 32UL, "file%d.txt", r + 1); }
            g.table.set_cell_text(r, 0, &buf[0]);
            unsafe { snprintf(&buf[0], 32UL, "%d KB", (r + 1) * 17 % 300); }
            g.table.set_cell_text(r, 1, &buf[0]);
            g.table.set_cell_text(r, 2, "Text");
        }
        case null: { }
    }
}

// ── the canvas widget inside the "Pickers" tab ────────────────────────────

static void draw_mini(&Widget w, &GuiCanvas c, ?&Gallery og) {
    float fw = (float)w.w; float fh = (float)w.h;
    struct GuiColor col = GuiColor::hex(0x228BE6U);
    match (og) { case some(g): { col = g.picker.get_color(); } case null: { } }
    struct GuiPaint p = GuiPaint::radial(fw * 0.35f, fh * 0.35f, fh * 0.8f);
    p.add_stop(0.0f, col.mix(GuiColor::hex(0xFFFFFFU), 0.6f));
    p.add_stop(1.0f, col);
    c.set_fill_paint(&p);
    c.fill_rounded_rect(0.0f, 0.0f, fw, fh, 8.0f);
    c.set_stroke_color(GuiColor::hex(0xFFFFFFU));
    c.set_line_width(3.0f);
    c.set_line_cap(GUI_CAP_ROUND);
    c.begin_path();
    c.move_to(12.0f, fh - 14.0f);
    c.cubic_to(fw * 0.3f, 6.0f, fw * 0.6f, fh, fw - 12.0f, 14.0f);
    c.stroke();
}

// ── the "Canvas" window ───────────────────────────────────────────────────

static void star(&GuiCanvas c, float cx, float cy, float r0, float r1, int points, float rot) {
    c.begin_path();
    int i = 0;
    while (i < points * 2) {
        float a = rot + (float)i * 3.14159265f / (float)points;
        float r = (i % 2 == 0) ? r1 : r0;
        float x = cx + r * cos_f(a);
        float y = cy + r * sin_f(a);
        if (i == 0) { c.move_to(x, y); } else { c.line_to(x, y); }
        i = i + 1;
    }
    c.close_path();
}

static void draw_canvas_window(&GuiApp app, &GuiWindow win, &GuiCanvas c, ?&Gallery og) {
    double t = 0.0;
    int dark = 0;
    struct GuiColor accent = GuiColor::hex(0x228BE6U);
    match (og) { case some(g): { t = g.t; dark = is_dark(g); accent = g.picker.get_color(); } case null: { } }
    float W = (float)win.logical_width();
    float H = (float)win.logical_height();
    c.scale((float)win.scale, (float)win.scale);              // draw in logical units
    struct GuiColor bg = dark ? GuiColor::hex(0x1E1F22U) : GuiColor::hex(0xF8F9FAU);
    struct GuiColor fg = dark ? GuiColor::hex(0xE9ECEFU) : GuiColor::hex(0x212529U);
    c.set_fill_color(bg);
    c.fill_rect(0.0f, 0.0f, W, H);

    // 1. A rotating star with a radial gradient and a round-joined outline.
    c.save();
    struct GuiPaint rp = GuiPaint::radial(110.0f, 110.0f, 90.0f);
    rp.add_stop(0.0f, GuiColor::hex(0xFFE066U));
    rp.add_stop(1.0f, GuiColor::hex(0xF76707U));
    c.set_fill_paint(&rp);
    star(c, 110.0f, 110.0f, 38.0f, 88.0f, 5, (float)(t * 0.6));
    c.fill();
    c.set_stroke_color(fg);
    c.set_line_width(4.0f);
    c.set_line_join(GUI_JOIN_ROUND);
    c.stroke();
    c.restore();

    // 2. Linear gradient card with a dashed border.
    c.save();
    struct GuiPaint lp = GuiPaint::linear(230.0f, 30.0f, 430.0f, 190.0f);
    lp.add_stop(0.0f, accent);
    lp.add_stop(1.0f, accent.shade(-0.5f));
    c.set_fill_paint(&lp);
    c.fill_rounded_rect(230.0f, 30.0f, 200.0f, 160.0f, 18.0f);
    float dash[2];
    dash[0] = 10.0f; dash[1] = 6.0f;
    c.set_dash(&dash[0], 2, (float)(t * 20.0));
    c.set_stroke_color(fg);
    c.set_line_width(2.0f);
    c.stroke_rounded_rect(222.0f, 22.0f, 216.0f, 176.0f, 24.0f);
    c.restore();
    c.set_fill_color(GuiColor::hex(0xFFFFFFU));
    c.set_font_size(20.0f);
    c.set_text_align(GUI_TEXT_CENTER);
    c.fill_text("Gradients", 330.0f, 105.0f);
    c.set_font_size(13.0f);
    c.fill_text("and dashed strokes", 330.0f, 128.0f);
    c.set_text_align(GUI_TEXT_LEFT);

    // 3. Blend modes: three overlapping circles, multiplied or screened.
    c.save();
    c.set_blend_mode(dark ? GUI_BLEND_SCREEN : GUI_BLEND_MULTIPLY);
    c.set_fill_color(GuiColor::hex(0xFF6B6BU)); c.fill_circle(80.0f, 280.0f, 50.0f);
    c.set_fill_color(GuiColor::hex(0x51CF66U)); c.fill_circle(130.0f, 280.0f, 50.0f);
    c.set_fill_color(GuiColor::hex(0x339AF0U)); c.fill_circle(105.0f, 322.0f, 50.0f);
    c.restore();

    // 4. Clipping: stripes clipped to a circle, plus caps and joins.
    c.save();
    c.begin_path();
    c.ellipse(320.0f, 300.0f, 70.0f, 70.0f);
    c.clip();
    int i = 0;
    while (i < 16) {
        c.set_fill_color(i % 2 == 0 ? accent : GuiColor::hex(0xFAB005U));
        c.fill_rect(240.0f + (float)i * 12.0f - (float)((int)(t * 24.0) % 24), 220.0f, 12.0f, 160.0f);
        i = i + 1;
    }
    c.restore();
    c.set_stroke_color(fg);
    c.set_line_width(2.0f);
    c.stroke_circle(320.0f, 300.0f, 70.0f);

    int k = 0;
    while (k < 3) {
        c.set_line_width(12.0f);
        c.set_line_cap(k);
        c.set_line_join(k);
        c.set_stroke_color(GuiColor::hex(0x845EF7U));
        c.begin_path();
        float y = 410.0f + (float)k * 0.0f;
        float x = 40.0f + (float)k * 140.0f;
        c.move_to(x, y + 40.0f);
        c.line_to(x + 40.0f, y);
        c.line_to(x + 80.0f, y + 40.0f);
        c.stroke();
        k = k + 1;
    }
    c.set_line_cap(GUI_CAP_BUTT);
    c.set_line_join(GUI_JOIN_MITER);

    // 5. Text: sizes, wrapped layout, measured underline.
    c.set_fill_color(fg);
    c.set_font_size(26.0f);
    c.fill_text("SafeC canvas", 30.0f, 510.0f);
    float tw = c.measure_text("SafeC canvas");
    c.set_stroke_color(accent);
    c.set_line_width(2.0f);
    c.draw_line(30.0f, 516.0f, 30.0f + tw, 516.0f);
    c.set_font_size(13.0f);
    c.fill_text("Anti-aliased paths, TrueType text, gradients, blend modes, clips.", 30.0f, 540.0f);
}

static int on_canvas_frame(&GuiApp app, &GuiWindow win, double now, ?&Gallery og) {
    match (og) { case some(g): { g.t = now / 1000.0; } case null: { } }
    return 1;
}

// The busy indicator and the progress bar move on a timer.
static void on_tick(&GuiApp app, unsigned int id, ?&Gallery og) {
    match (og) {
        case some(g): {
            double v = g.busy.value() + 0.01;
            if (v > 1.0) { v = 0.0; }
            g.busy.set_value(v);
        }
        case null: { }
    }
}

// ── the "Widgets" window ──────────────────────────────────────────────────

static &Widget build_menu(&Gallery g) {
    &Widget mb = Widget::menubar();
    &Widget file = Widget::menu("File");
    file.add_child(Widget::menu_item("Settings...").set_shortcut("Ctrl+,").on_click(on_settings, g));
    file.add_child(Widget::menu_item("Message box...").set_shortcut("Ctrl+M").on_click(on_message_box, g));
    file.add_child(Widget::menu_separator());
    &Widget recent = Widget::menu("Open recent");
    recent.add_child(Widget::menu_item("notes.txt").on_click(on_button, g));
    recent.add_child(Widget::menu_item("photo.png").on_click(on_button, g));
    file.add_child(recent);
    file.add_child(Widget::menu_separator());
    file.add_child(Widget::menu_item("Quit").set_shortcut("Ctrl+Q").on_click(on_quit, g));
    mb.add_child(file);
    &Widget edit = Widget::menu("Edit");
    edit.add_child(Widget::menu_item("Undo").set_shortcut("Ctrl+Z").set_enabled(0));
    edit.add_child(Widget::menu_item("Add a table row").set_shortcut("Ctrl+R").on_click(on_add_row, g));
    mb.add_child(edit);
    &Widget view = Widget::menu("View");
    g.dark = Widget::menu_item("Dark theme").set_shortcut("Ctrl+D").on_change(on_dark, g);
    g.dark.set_checked(GuiTheme::system().dark);
    view.add_child(g.dark);
    mb.add_child(view);
    return mb;
}

static &Widget build_controls(&Gallery g) {
    &Widget page = Widget::hstack().set_padding(12).set_spacing(12);
    &Widget left = Widget::vstack().set_stretch(1);

    &Widget buttons = Widget::group("Buttons");
    &Widget row = Widget::hstack();
    row.add_child(Widget::button("Default").on_click(on_button, g).set_tooltip("A plain push button"));
    row.add_child(Widget::button("Another").on_click(on_button, g));
    row.add_child(Widget::button("Disabled").set_enabled(0));
    buttons.add_child(row);
    buttons.add_child(Widget::checkbox("A check box", 1).set_tooltip("Space toggles it when focused"));
    buttons.add_child(Widget::checkbox("Unchecked", 0));
    buttons.add_child(Widget::toggle("A toggle switch", 1));
    &Widget radios = Widget::radio_group();
    radios.add_item("First choice"); radios.add_item("Second choice"); radios.add_item("Third choice");
    radios.set_selected(0);
    radios.on_change(on_select, g);
    buttons.add_child(radios);
    left.add_child(buttons);

    &Widget values = Widget::group("Values");
    &Widget srow = Widget::hstack();
    g.slider = Widget::slider(0.0, 100.0, 40.0).set_stretch(1).on_change(on_slider, g);
    g.sliderLabel = Widget::label("40 %").set_min_size(48, 0);
    srow.add_child(g.slider);
    srow.add_child(g.sliderLabel);
    values.add_child(srow);
    g.progress = Widget::progress(0.4);
    values.add_child(g.progress);
    g.busy = Widget::progress(0.0);
    values.add_child(g.busy);
    values.add_child(Widget::progress(0.0).set_indeterminate(1).set_tooltip("Indeterminate: busy"));
    &Widget vrow = Widget::hstack();
    vrow.add_child(Widget::spinner(0.0, 100.0, 12.5, 0.5).set_decimals(1));
    vrow.add_child(Widget::slider(0.0, 10.0, 3.0).set_orientation(WIDGET_VERTICAL).set_pref_size(24, 80));
    values.add_child(vrow);
    left.add_child(values);
    page.add_child(left);

    &Widget right = Widget::vstack().set_stretch(1);
    &Widget text = Widget::group("Text");
    g.input = Widget::textinput("Type something").on_change(on_input, g);
    text.add_child(g.input);
    text.add_child(Widget::textinput("Password").set_password(1));
    text.add_child(Widget::textinput("").set_text("Read-only text").set_read_only(1));
    &Widget combo = Widget::combo().on_change(on_select, g);
    combo.add_item("Apples"); combo.add_item("Bananas"); combo.add_item("Cherries"); combo.add_item("Dates");
    text.add_child(combo);
    &Widget area = Widget::textarea().set_pref_size(0, 110).set_wrap(1);
    area.set_text("A multi-line text area.\nIt wraps long lines, scrolls, selects with the mouse or Shift+arrows, and copies and pastes through the system clipboard.");
    text.add_child(area);
    right.add_child(text);
    &Widget labels = Widget::group("Labels");
    labels.add_child(Widget::label("A label that wraps its text onto several lines when the window is narrow enough.").set_wrap(1));
    labels.add_child(Widget::label("Bold, larger").set_bold(1).set_font_size(17.0f));
    labels.add_child(Widget::separator());
    labels.add_child(Widget::label("Centered").set_text_align(GUI_TEXT_CENTER));
    right.add_child(labels);
    page.add_child(right);
    return page;
}

static &Widget build_data(&Gallery g) {
    &Widget split = Widget::splitter(WIDGET_HORIZONTAL);
    split.set_value(0.32);
    &Widget lst = Widget::list().on_change(on_select, g);
    int i = 0;
    while (i < 40) {
        char buf[32];
        unsafe { snprintf(&buf[0], 32UL, "List item %d", i + 1); }
        lst.add_item(&buf[0]);
        i = i + 1;
    }
    &Widget ctx = Widget::menu("");
    ctx.add_child(Widget::menu_item("Copy").set_shortcut("Ctrl+C").on_click(on_button, g));
    ctx.add_child(Widget::menu_item("Delete").set_shortcut("Delete").on_click(on_button, g));
    lst.set_context_menu(ctx);
    lst.set_tooltip("Right-click for a context menu");
    split.add_child(lst);

    &Widget vsplit = Widget::splitter(WIDGET_VERTICAL);
    &Widget tree = Widget::tree().on_change(on_select, g);
    int docs = tree.add_node(-1, "Documents");
    tree.add_node(docs, "report.pdf");
    int pics = tree.add_node(docs, "Pictures");
    tree.add_node(pics, "beach.png");
    tree.add_node(pics, "mountain.jpg");
    tree.add_node(docs, "notes.txt");
    int music = tree.add_node(-1, "Music");
    tree.add_node(music, "song.ogg");
    tree.set_expanded(docs, 1);
    vsplit.add_child(tree);
    g.table = Widget::table().on_change(on_select, g);
    g.table.add_column("Name", 160);
    g.table.add_column("Size", 90);
    g.table.add_column("Kind", 0);
    i = 0;
    while (i < 12) { on_add_row(g.table, g); i = i + 1; }
    vsplit.add_child(g.table);
    split.add_child(vsplit);

    &Widget page = Widget::vstack().set_padding(12);
    page.add_child(split.set_stretch(1));
    &Widget row = Widget::hstack();
    row.add_child(Widget::button("Add a row").on_click(on_add_row, g));
    row.add_child(Widget::label("Drag the dividers; drag column edges in the header."));
    page.add_child(row);
    return page;
}

static &Widget build_pickers(&Gallery g) {
    &Widget page = Widget::hstack().set_padding(12).set_spacing(12);
    &Widget col = Widget::group("Color picker");
    g.picker = Widget::color_picker(GuiColor::hex(0x228BE6U)).on_change(on_color, g);
    col.add_child(g.picker);
    page.add_child(col);
    &Widget mid = Widget::vstack().set_stretch(1);
    &Widget img = Widget::group("Image view");
    img.add_child(Widget::image(&g.photo).set_pref_size(0, 140).set_fit(WIDGET_FIT_CONTAIN));
    mid.add_child(img);
    &Widget cv = Widget::group("Canvas widget");
    cv.add_child(Widget::canvas(draw_mini, g).set_pref_size(0, 120));
    mid.add_child(cv);
    page.add_child(mid);
    return page;
}

static &Widget build_layouts(&Gallery g) {
    &Widget page = Widget::hstack().set_padding(12).set_spacing(12);
    &Widget gridBox = Widget::group("Grid");
    &Widget grid = Widget::grid(3);
    grid.add_child(Widget::label("Name"));
    grid.add_child(Widget::textinput("First").set_stretch(1));
    grid.add_child(Widget::textinput("Last"));
    grid.add_child(Widget::label("Email"));
    grid.add_child(Widget::textinput("name@example.org").set_span(1, 2));
    grid.add_child(Widget::label("Notes"));
    grid.add_child(Widget::textarea().set_span(1, 2).set_pref_size(0, 60));
    grid.set_column_stretch(1, 1);
    grid.set_column_stretch(2, 1);
    gridBox.add_child(grid);
    &Widget stackBox = Widget::group("Stack (overlay)");
    &Widget st = Widget::stack().set_pref_size(0, 80);
    st.add_child(Widget::canvas(draw_mini, g));
    st.add_child(Widget::label("On top").set_bold(1).set_align2(WIDGET_ALIGN_CENTER, WIDGET_ALIGN_CENTER));
    stackBox.add_child(st);
    &Widget leftCol = Widget::vstack().set_stretch(1);
    leftCol.add_child(gridBox);
    leftCol.add_child(stackBox);
    page.add_child(leftCol);

    &Widget scrollBox = Widget::group("Scroll view").set_stretch(1);
    &Widget sc = Widget::scroll().set_stretch(1).set_pref_size(0, 240);
    &Widget content = Widget::vstack().set_padding(4);
    int i = 0;
    while (i < 30) {
        char buf[48];
        unsafe { snprintf(&buf[0], 48UL, "Row %d of a long scrolled column", i + 1); }
        &Widget r = Widget::hstack();
        r.add_child(Widget::checkbox(&buf[0], i % 3 == 0).set_stretch(1));
        r.add_child(Widget::button("Go").on_click(on_button, g));
        content.add_child(r);
        i = i + 1;
    }
    sc.add_child(content);
    scrollBox.add_child(sc);
    page.add_child(scrollBox);
    return page;
}

static &Widget build_widgets_window(&Gallery g) {
    &Widget root = Widget::vstack().set_spacing(0);
    root.add_child(build_menu(g));
    &Widget tabs = Widget::tabs().set_stretch(1).set_margin(8).set_name("tabs");
    tabs.add_tab("Controls", build_controls(g));
    tabs.add_tab("Data", build_data(g));
    tabs.add_tab("Pickers", build_pickers(g));
    tabs.add_tab("Layouts", build_layouts(g));
    root.add_child(tabs);
    &Widget sb = Widget::statusbar();
    g.status = Widget::label("Ready. Try the menus, Tab and Shift+Tab, right-clicks and tooltips.").set_stretch(1);
    sb.add_child(g.status);
    sb.add_child(Widget::separator());
    sb.add_child(Widget::button("Settings...").on_click(on_settings, g));
    sb.add_child(Widget::button("Message box...").on_click(on_message_box, g));
    root.add_child(sb);
    return root;
}

static ?&Widget ui_root_find(&GuiApp app, &GuiWindow win, const &char name) {
    match (app.ui_of(win)) { case some(ui): { return ui.root.find(name); } case null: { } }
    return null;
}

// A procedural "photo" for the image view.
static struct GuiImage make_photo() {
    struct GuiImage img = GuiImage::new(240, 160);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    struct GuiPaint sky = GuiPaint::linear(0.0f, 0.0f, 0.0f, 160.0f);
    sky.add_stop(0.0f, GuiColor::hex(0x4DABF7U));
    sky.add_stop(1.0f, GuiColor::hex(0xFFD8A8U));
    c.set_fill_paint(&sky);
    c.fill_rect(0.0f, 0.0f, 240.0f, 160.0f);
    c.set_fill_color(GuiColor::hex(0xFFF3BFU));
    c.fill_circle(180.0f, 50.0f, 22.0f);
    c.set_fill_color(GuiColor::hex(0x2F9E44U));
    c.begin_path();
    c.move_to(0.0f, 160.0f); c.line_to(0.0f, 110.0f);
    c.quad_to(60.0f, 60.0f, 120.0f, 112.0f);
    c.quad_to(180.0f, 80.0f, 240.0f, 105.0f);
    c.line_to(240.0f, 160.0f);
    c.close_path();
    c.fill();
    c.free();
    return img;
}

int main(int argc, char** argv) {
    const char* snapshot = (const char*)0;
    unsafe {
        if (argc >= 3 && strcmp(argv[1], "--snapshot") == 0) {
            snapshot = argv[2];
            gui_use_headless();
        }
    }
    struct GuiApp app = GuiApp::new();
    struct Gallery g;
    g.app = &app;
    g.photo = make_photo();
    g.t = 0.0;
    g.clicks = 0;

    struct GuiWindowOptions wo = GuiWindowOptions::new("SafeC GUI gallery — widgets", 860, 620);
    wo.minWidth = 560; wo.minHeight = 420;
    g.widgets = app.open(&wo);
    struct GuiWindowOptions co = GuiWindowOptions::new("SafeC GUI gallery — canvas", 470, 560);
    co.minWidth = 300; co.minHeight = 300;
    g.canvas = app.open(&co);
    if (g.widgets.is_null() || g.canvas.is_null()) {
        unsafe { printf("gui_gallery: no window (no display?)\n"); }
        app.free();
        return 0;
    }
    &GuiWindow ww;
    &GuiWindow cw;
    unsafe { ww = (struct GuiWindow*)g.widgets; cw = (struct GuiWindow*)g.canvas; }

    app.set_root(ww, build_widgets_window(&g));
    g.settings = make_settings_dialog(&g);
    apply_theme(&g, g.dark.is_checked());
    app.on_draw(cw, draw_canvas_window, &g);
    app.on_frame(cw, on_canvas_frame, &g);
    app.set_interval(50.0, on_tick, &g);

    if (snapshot != (const char*)0) {
        int i = 0;
        while (i < 3) { app.pump(0); i = i + 1; }
        char path[512];
        struct GuiImage b = GuiImage::from_window(cw);
        unsafe {
            snprintf(&path[0], 512UL, "%s/canvas.png", snapshot);
            b.save_png(&path[0]);
        }
        b.free();
        // each tab of the widgets window: widgets.png, widgets-2.png, ...
        ?&Widget otabs = ui_root_find(&app, ww, "tabs");
        int page = 0;
        while (page < 4) {
            match (otabs) { case some(t): { t.set_selected(page); } case null: { } }
            app.pump(0);
            struct GuiImage a = GuiImage::from_window(ww);
            unsafe {
                if (page == 0) { snprintf(&path[0], 512UL, "%s/widgets.png", snapshot); }
                else { snprintf(&path[0], 512UL, "%s/widgets-%d.png", snapshot, page + 1); }
                a.save_png(&path[0]);
            }
            a.free();
            page = page + 1;
        }
        unsafe { printf("gui_gallery: wrote the snapshots to %s\n", snapshot); }
    } else {
        app.run();
    }
    app.free();
    g.settings.free();
    g.photo.free();
    return 0;
}
