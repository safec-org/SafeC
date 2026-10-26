// std::gui vector canvas, headless (into GuiImages): coverage accuracy,
// fill rules, strokes (width, caps, joins, dashes), gradients, patterns,
// transforms, save/restore, clipping, blend modes, image drawing, paths,
// and a golden image (tests/stdlib/fixtures/gui/canvas_scene.png).
#include <std/test/test.h>
#include <std/mem.h>
#include <std/math.h>
#include <std/gui/gui.h>
#include <std/gui/gui_canvas.h>
#include <std/gui/gui_path.h>
#include <std/gui/gui_png.h>
#include <std/gui/gui_text.h>

static struct GuiColor rgb(unsigned int hex) { return GuiColor::hex(hex); }

// channel c (0=R .. 3=A) of pixel (x, y), as stored (premultiplied)
static int ch(&GuiImage img, int x, int y, int c) {
    unsigned int v = img.get_pixel(x, y);
    return (int)((v >> (unsigned int)(24 - c * 8)) & 0xFFU);
}

static int near(int a, int b, int tol) { return (a - b <= tol && b - a <= tol) ? 1 : 0; }

static double alpha_sum(&GuiImage img) {
    double s = 0.0;
    int y = 0;
    while (y < img.height) {
        int x = 0;
        while (x < img.width) { s = s + (double)ch(img, x, y, 3) / 255.0; x = x + 1; }
        y = y + 1;
    }
    return s;
}

static void wipe(&GuiImage img) {
    unsafe { safe_memset(((unsigned char*)img.pixels)[0..(unsigned long)(img.width * img.height * 4)], 0); }
}

void t_rects() {
    struct GuiImage img = GuiImage::new(40, 30);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.set_fill_color(rgb(0xFF0000U));
    c.fill_rect(5.0f, 5.0f, 10.0f, 8.0f);              // integer: exact
    ASSERT_EQ(ch(&img, 5, 5, 0), 255);
    ASSERT_EQ(ch(&img, 14, 12, 3), 255);
    ASSERT_EQ(ch(&img, 15, 12, 3), 0);
    ASSERT_EQ(ch(&img, 4, 5, 3), 0);
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 80, 0));
    wipe(&img);
    c.fill_rect(10.5f, 10.0f, 10.0f, 4.0f);           // half-pixel edges
    ASSERT_TRUE(near(ch(&img, 10, 11, 3), 128, 2));
    ASSERT_EQ(ch(&img, 11, 11, 3), 255);
    ASSERT_TRUE(near(ch(&img, 20, 11, 3), 128, 2));
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 40, 1));
    wipe(&img);
    c.fill_rect(10.0f, 10.25f, 4.0f, 2.0f);           // quarter-pixel vertical edges
    ASSERT_TRUE(near(ch(&img, 11, 10, 3), 191, 3));
    ASSERT_TRUE(near(ch(&img, 11, 12, 3), 64, 3));
    c.free();
    img.free();
}

void t_coverage_circle() {
    struct GuiImage img = GuiImage::new(100, 100);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.set_fill_color(rgb(0x000000U));
    c.fill_circle(50.3f, 49.6f, 30.0f);
    double area = alpha_sum(&img);
    double exact = 3.14159265358979 * 900.0;
    ASSERT_TRUE(area > exact * 0.997 && area < exact * 1.003);
    ASSERT_EQ(ch(&img, 50, 50, 3), 255);
    ASSERT_EQ(ch(&img, 5, 5, 3), 0);
    // An anti-aliased edge: some pixels strictly between 0 and 255.
    int partial = 0;
    int x = 0;
    while (x < 100) { int a = ch(&img, x, 50, 3); if (a > 10 && a < 245) partial = partial + 1; x = x + 1; }
    ASSERT_TRUE(partial >= 2);
    // Aliased: none.
    wipe(&img);
    c.set_anti_alias(0);
    c.fill_circle(50.3f, 49.6f, 30.0f);
    partial = 0;
    int y = 0;
    while (y < 100) { x = 0; while (x < 100) { int a = ch(&img, x, y, 3); if (a != 0 && a != 255) partial = partial + 1; x = x + 1; } y = y + 1; }
    ASSERT_EQ(partial, 0);
    c.free();
    img.free();
}

void t_fill_rules() {
    struct GuiImage img = GuiImage::new(60, 60);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    struct GuiPath p = GuiPath::new();
    p.rect(5.0f, 5.0f, 50.0f, 50.0f);
    p.rect(20.0f, 20.0f, 20.0f, 20.0f);               // same winding: nonzero fills it
    c.set_fill_color(rgb(0x0000FFU));
    c.fill_path(&p);
    ASSERT_EQ(ch(&img, 30, 30, 3), 255);
    wipe(&img);
    c.set_fill_rule(GUI_FILL_EVENODD);
    c.fill_path(&p);
    ASSERT_EQ(ch(&img, 30, 30, 3), 0);               // even-odd: a hole
    ASSERT_EQ(ch(&img, 10, 10, 3), 255);
    // A hole by opposite winding works for nonzero too.
    p.clear();
    p.rect(5.0f, 5.0f, 50.0f, 50.0f);
    p.move_to(20.0f, 20.0f); p.line_to(20.0f, 40.0f); p.line_to(40.0f, 40.0f); p.line_to(40.0f, 20.0f); p.close();
    wipe(&img);
    c.set_fill_rule(GUI_FILL_NONZERO);
    c.fill_path(&p);
    ASSERT_EQ(ch(&img, 30, 30, 3), 0);
    // A pentagram: the center is wound twice.
    p.clear();
    float pts[10];
    int i = 0;
    while (i < 5) {
        float a = -1.5707963f + (float)(i * 2) * 1.2566371f;
        pts[i * 2] = 30.0f + 25.0f * cos_f(a);
        pts[i * 2 + 1] = 30.0f + 25.0f * sin_f(a);
        i = i + 1;
    }
    p.polygon(&pts[0], 5);
    wipe(&img);
    c.fill_path(&p);
    ASSERT_EQ(ch(&img, 30, 31, 3), 255);
    wipe(&img);
    c.set_fill_rule(GUI_FILL_EVENODD);
    c.fill_path(&p);
    ASSERT_EQ(ch(&img, 30, 31, 3), 0);
    p.free();
    c.free();
    img.free();
}

void t_strokes() {
    struct GuiImage img = GuiImage::new(100, 60);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.set_stroke_color(rgb(0x000000U));
    c.set_line_width(4.0f);
    c.draw_line(10.0f, 20.0f, 90.0f, 20.0f);          // covers rows 18..21, x 10..89
    ASSERT_EQ(ch(&img, 50, 18, 3), 255);
    ASSERT_EQ(ch(&img, 50, 21, 3), 255);
    ASSERT_EQ(ch(&img, 50, 17, 3), 0);
    ASSERT_EQ(ch(&img, 50, 22, 3), 0);
    ASSERT_EQ(ch(&img, 9, 20, 3), 0);                 // butt cap
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 320, 1));
    wipe(&img);
    c.set_line_cap(GUI_CAP_SQUARE);
    c.draw_line(10.0f, 20.0f, 90.0f, 20.0f);
    ASSERT_EQ(ch(&img, 8, 20, 3), 255);               // extends by half the width
    ASSERT_EQ(ch(&img, 7, 20, 3), 0);
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 336, 1));
    wipe(&img);
    c.set_line_cap(GUI_CAP_ROUND);
    c.draw_line(10.0f, 20.0f, 90.0f, 20.0f);
    double expect = 320.0 + 3.14159265 * 4.0;         // plus two half discs of r = 2
    ASSERT_TRUE(alpha_sum(&img) > expect - 1.0 && alpha_sum(&img) < expect + 1.0);

    // Joins on a right angle corner at (50, 30).
    struct GuiPath p = GuiPath::new();
    p.move_to(20.0f, 30.0f); p.line_to(50.0f, 30.0f); p.line_to(50.0f, 55.0f);
    c.set_line_cap(GUI_CAP_BUTT);
    c.set_line_width(8.0f);
    wipe(&img);
    c.set_line_join(GUI_JOIN_MITER);
    c.stroke_path(&p);
    ASSERT_EQ(ch(&img, 53, 27, 3), 255);              // the miter fills the outer corner
    wipe(&img);
    c.set_line_join(GUI_JOIN_BEVEL);
    c.stroke_path(&p);
    ASSERT_EQ(ch(&img, 53, 27, 3), 0);                // cut off
    ASSERT_EQ(ch(&img, 51, 29, 3), 255);
    wipe(&img);
    c.set_line_join(GUI_JOIN_ROUND);
    c.stroke_path(&p);
    ASSERT_TRUE(ch(&img, 53, 27, 3) > 0 && ch(&img, 53, 27, 3) < 255);   // on the arc
    // No double blending where segments overlap (a translucent stroke).
    wipe(&img);
    c.set_stroke_color(GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)128));
    c.stroke_path(&p);
    ASSERT_TRUE(near(ch(&img, 50, 30, 3), 128, 1));
    ASSERT_TRUE(near(ch(&img, 35, 30, 3), 128, 1));

    // Dashes: 10 on, 10 off along a horizontal line.
    wipe(&img);
    c.set_stroke_color(rgb(0x000000U));
    c.set_line_width(2.0f);
    float dash[2];
    dash[0] = 10.0f; dash[1] = 10.0f;
    c.set_dash(&dash[0], 2, 0.0f);
    c.draw_line(0.0f, 10.0f, 100.0f, 10.0f);
    ASSERT_EQ(ch(&img, 5, 10, 3), 255);
    ASSERT_EQ(ch(&img, 15, 10, 3), 0);
    ASSERT_EQ(ch(&img, 25, 10, 3), 255);
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 100, 1));
    c.set_dash(&dash[0], 2, 5.0f);                     // offset shifts the pattern
    wipe(&img);
    c.draw_line(0.0f, 10.0f, 100.0f, 10.0f);
    ASSERT_EQ(ch(&img, 2, 10, 3), 255);
    ASSERT_EQ(ch(&img, 7, 10, 3), 0);
    c.set_dash(&dash[0], 0, 0.0f);

    // stroke_to_path gives the same outline as a path.
    struct GuiPath out = GuiPath::new();
    c.set_line_width(4.0f);
    p.clear();
    p.move_to(10.0f, 40.0f); p.line_to(60.0f, 40.0f);
    c.stroke_to_path(&p, &out);
    struct GuiRect b = out.bounds();
    ASSERT_TRUE(b.x > 9.9f && b.x < 10.1f && b.y > 37.9f && b.y < 38.1f && b.w > 49.9f && b.h > 3.9f);
    out.free();
    p.free();
    c.free();
    img.free();
}

void t_gradients() {
    struct GuiImage img = GuiImage::new(101, 20);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    struct GuiPaint g = GuiPaint::linear(0.0f, 0.0f, 101.0f, 0.0f);
    g.add_stop(1.0f, rgb(0x0000FFU));                  // any order
    g.add_stop(0.0f, rgb(0xFF0000U));
    c.set_fill_paint(&g);
    c.fill_rect(0.0f, 0.0f, 101.0f, 20.0f);
    ASSERT_TRUE(ch(&img, 0, 5, 0) > 250 && ch(&img, 0, 5, 2) < 5);
    ASSERT_TRUE(ch(&img, 100, 5, 2) > 250 && ch(&img, 100, 5, 0) < 5);
    ASSERT_TRUE(near(ch(&img, 50, 5, 0), 128, 3) && near(ch(&img, 50, 5, 2), 128, 3));
    // Spread: repeat restarts the ramp, reflect mirrors it.
    struct GuiPaint r = GuiPaint::linear(0.0f, 0.0f, 50.0f, 0.0f);
    r.add_stop(0.0f, rgb(0x000000U));
    r.add_stop(1.0f, rgb(0xFFFFFFU));
    r.set_spread(GUI_SPREAD_REPEAT);
    c.set_fill_paint(&r);
    c.fill_rect(0.0f, 0.0f, 101.0f, 20.0f);
    ASSERT_TRUE(ch(&img, 52, 5, 0) < 20);
    r.set_spread(GUI_SPREAD_REFLECT);
    c.set_fill_paint(&r);
    c.fill_rect(0.0f, 0.0f, 101.0f, 20.0f);
    ASSERT_TRUE(ch(&img, 52, 5, 0) > 235);
    r.set_spread(GUI_SPREAD_PAD);
    c.set_fill_paint(&r);
    c.fill_rect(0.0f, 0.0f, 101.0f, 20.0f);
    ASSERT_TRUE(ch(&img, 90, 5, 0) > 250);
    c.free();
    img.free();

    struct GuiImage im2 = GuiImage::new(60, 60);
    struct GuiCanvas c2 = GuiCanvas::for_image(&im2);
    struct GuiPaint rad = GuiPaint::radial(30.0f, 30.0f, 25.0f);
    rad.add_stop(0.0f, rgb(0xFFFFFFU));
    rad.add_stop(1.0f, rgb(0x000000U));
    c2.set_fill_paint(&rad);
    c2.fill_rect(0.0f, 0.0f, 60.0f, 60.0f);
    ASSERT_TRUE(ch(&im2, 30, 30, 0) > 245);
    ASSERT_TRUE(ch(&im2, 30, 4, 0) < 15);
    ASSERT_TRUE(near(ch(&im2, 42, 30, 0), ch(&im2, 30, 42, 0), 2));   // symmetric
    ASSERT_TRUE(near(ch(&im2, 42, 30, 0), 128, 14));
    struct GuiPaint sw = GuiPaint::sweep(30.0f, 30.0f, 0.0f);
    sw.add_stop(0.0f, rgb(0xFF0000U));
    sw.add_stop(0.5f, rgb(0x00FF00U));
    sw.add_stop(1.0f, rgb(0xFF0000U));
    c2.set_fill_paint(&sw);
    c2.fill_rect(0.0f, 0.0f, 60.0f, 60.0f);
    ASSERT_TRUE(ch(&im2, 10, 31, 1) > 200);            // angle pi: the green stop
    ASSERT_TRUE(ch(&im2, 55, 31, 0) > 200);
    // Paint opacity and global alpha.
    wipe(&im2);
    struct GuiPaint half = GuiPaint::solid(rgb(0xFF0000U));
    half.set_opacity(0.5f);
    c2.set_fill_paint(&half);
    c2.set_global_alpha(0.5f);
    c2.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_TRUE(near(ch(&im2, 5, 5, 3), 64, 1));
    c2.free();
    im2.free();
}

void t_transform_state_clip() {
    struct GuiImage img = GuiImage::new(80, 80);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.set_fill_color(rgb(0x00FF00U));
    c.save();
    c.translate(40.0f, 40.0f);
    c.rotate(1.5707963f);                              // 90 degrees: (x, y) -> (-y, x)
    c.fill_rect(0.0f, 0.0f, 20.0f, 6.0f);              // lands at x 34..40, y 40..60
    c.restore();
    ASSERT_EQ(ch(&img, 37, 50, 3), 255);
    ASSERT_EQ(ch(&img, 45, 42, 3), 0);
    struct GuiMatrix m = c.get_transform();
    ASSERT_TRUE(m.is_identity());
    wipe(&img);
    c.save();
    c.scale(2.0f, 3.0f);
    c.fill_rect(1.0f, 1.0f, 5.0f, 5.0f);               // x 2..12, y 3..18
    c.restore();
    ASSERT_EQ(ch(&img, 11, 17, 3), 255);
    ASSERT_EQ(ch(&img, 12, 17, 3), 0);
    ASSERT_EQ(ch(&img, 11, 18, 3), 0);

    // Rectangle clip, restored by restore().
    wipe(&img);
    c.save();
    c.clip_rect(10.0f, 10.0f, 20.0f, 20.0f);
    c.fill_rect(0.0f, 0.0f, 80.0f, 80.0f);
    c.restore();
    ASSERT_EQ(ch(&img, 15, 15, 3), 255);
    ASSERT_EQ(ch(&img, 5, 15, 3), 0);
    ASSERT_EQ(ch(&img, 30, 15, 3), 0);
    ASSERT_TRUE(near((int)(alpha_sum(&img) + 0.5), 400, 0));
    c.fill_rect(0.0f, 0.0f, 1.0f, 1.0f);               // clip gone
    ASSERT_EQ(ch(&img, 0, 0, 3), 255);

    // Path clip (a circle) intersected with a rect clip.
    wipe(&img);
    c.save();
    struct GuiPath circle = GuiPath::new();
    circle.circle(40.0f, 40.0f, 20.0f);
    c.clip_path(&circle);
    c.clip_rect(0.0f, 0.0f, 40.0f, 80.0f);              // left half only
    c.fill_rect(0.0f, 0.0f, 80.0f, 80.0f);
    c.restore();
    ASSERT_EQ(ch(&img, 30, 40, 3), 255);
    ASSERT_EQ(ch(&img, 50, 40, 3), 0);
    ASSERT_EQ(ch(&img, 22, 22, 3), 0);
    double area = alpha_sum(&img);
    ASSERT_TRUE(area > 628.3 - 3.0 && area < 628.3 + 3.0);   // half of pi * 400
    // Nested saves keep their own clips.
    wipe(&img);
    c.save();
    c.clip_rect(0.0f, 0.0f, 40.0f, 40.0f);
    c.save();
    c.clip_path(&circle);
    c.restore();
    c.fill_rect(0.0f, 0.0f, 80.0f, 80.0f);
    c.restore();
    ASSERT_EQ(ch(&img, 1, 1, 3), 255);                 // the circle clip was popped
    ASSERT_EQ(ch(&img, 41, 1, 3), 0);                  // the outer one still applied
    circle.free();
    c.free();
    img.free();
}

void t_blend() {
    struct GuiImage img = GuiImage::new(10, 10);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    c.clear(GuiColor::rgb((unsigned char)200, (unsigned char)100, (unsigned char)50));
    c.set_blend_mode(GUI_BLEND_MULTIPLY);
    c.set_fill_color(GuiColor::rgb((unsigned char)128, (unsigned char)255, (unsigned char)0));
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_TRUE(near(ch(&img, 3, 3, 0), 100, 1));
    ASSERT_TRUE(near(ch(&img, 3, 3, 1), 100, 1));
    ASSERT_EQ(ch(&img, 3, 3, 2), 0);
    c.clear(GuiColor::rgb((unsigned char)200, (unsigned char)100, (unsigned char)50));
    c.set_blend_mode(GUI_BLEND_SCREEN);
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_TRUE(near(ch(&img, 3, 3, 0), 228, 1));      // 200 + 128 - 200*128/255
    ASSERT_EQ(ch(&img, 3, 3, 1), 255);
    c.clear(GuiColor::rgb((unsigned char)200, (unsigned char)100, (unsigned char)50));
    c.set_blend_mode(GUI_BLEND_DIFFERENCE);
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_TRUE(near(ch(&img, 3, 3, 0), 72, 1));
    ASSERT_TRUE(near(ch(&img, 3, 3, 1), 155, 1));
    // Porter-Duff: src replaces even with alpha; clear erases; dst-out
    // punches by source alpha.
    c.set_blend_mode(GUI_BLEND_SRC);
    c.set_fill_color(GuiColor::rgba((unsigned char)255, (unsigned char)0, (unsigned char)0, (unsigned char)128));
    c.fill_rect(0.0f, 0.0f, 5.0f, 10.0f);
    ASSERT_TRUE(near(ch(&img, 1, 1, 3), 128, 1) && near(ch(&img, 1, 1, 0), 128, 1));
    c.set_blend_mode(GUI_BLEND_CLEAR);
    c.fill_rect(0.0f, 0.0f, 2.0f, 2.0f);
    ASSERT_EQ(ch(&img, 1, 1, 3), 0);
    c.set_blend_mode(GUI_BLEND_SRC_OVER);
    c.clear(GuiColor::rgb((unsigned char)0, (unsigned char)0, (unsigned char)255));
    c.set_blend_mode(GUI_BLEND_DST_OUT);
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_TRUE(near(ch(&img, 5, 5, 3), 127, 1));
    c.set_blend_mode(GUI_BLEND_DST_OVER);                 // behind: only fills what's transparent
    c.clear(GuiColor::rgba((unsigned char)0, (unsigned char)0, (unsigned char)0, (unsigned char)0));
    c.set_fill_color(rgb(0x00FF00U));
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_EQ(ch(&img, 5, 5, 1), 255);
    c.set_fill_color(rgb(0xFF0000U));
    c.fill_rect(0.0f, 0.0f, 10.0f, 10.0f);
    ASSERT_EQ(ch(&img, 5, 5, 0), 0);
    // clear_rect
    c.set_blend_mode(GUI_BLEND_SRC_OVER);
    c.clear_rect(0.0f, 0.0f, 3.0f, 3.0f);
    ASSERT_EQ(ch(&img, 1, 1, 3), 0);
    ASSERT_EQ(ch(&img, 4, 4, 3), 255);
    struct GuiColor back = c.get_pixel(5, 5);
    ASSERT_EQ(back.g, 255);
    c.free();
    img.free();
}

void t_images() {
    struct GuiImage src = GuiImage::new(4, 4);
    int y = 0;
    while (y < 4) {
        int x = 0;
        while (x < 4) {
            src.set_pixel(x, y, ((x + y) % 2) == 0 ? 0xFF0000FFU : 0x0000FFFFU);
            x = x + 1;
        }
        y = y + 1;
    }
    struct GuiImage dst = GuiImage::new(20, 20);
    struct GuiCanvas c = GuiCanvas::for_image(&dst);
    c.draw_image(&src, 3.0f, 2.0f);                      // exact copy
    ASSERT_EQ(dst.get_pixel(3, 2), 0xFF0000FFU);
    ASSERT_EQ(dst.get_pixel(4, 2), 0x0000FFFFU);
    ASSERT_EQ(dst.get_pixel(6, 5), 0xFF0000FFU);
    ASSERT_EQ(ch(&dst, 7, 2, 3), 0);
    wipe(&dst);
    c.set_image_filter(GUI_FILTER_NEAREST);
    c.draw_image_scaled(&src, 0.0f, 0.0f, 16.0f, 16.0f);  // 4x, blocky
    ASSERT_EQ(dst.get_pixel(1, 1), 0xFF0000FFU);
    ASSERT_EQ(dst.get_pixel(5, 1), 0x0000FFFFU);
    ASSERT_EQ(dst.get_pixel(3, 3), 0xFF0000FFU);
    wipe(&dst);
    c.set_image_filter(GUI_FILTER_BILINEAR);
    c.draw_image_scaled(&src, 0.0f, 0.0f, 16.0f, 16.0f);  // smooth at the cell borders
    ASSERT_TRUE(ch(&dst, 4, 1, 0) > 40 && ch(&dst, 4, 1, 0) < 215);
    wipe(&dst);
    c.draw_image_rect(&src, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 5.0f, 5.0f);   // one source pixel, blown up
    ASSERT_EQ(dst.get_pixel(2, 2), 0x0000FFFFU);
    ASSERT_EQ(ch(&dst, 6, 2, 3), 0);
    // A pattern paint tiles.
    wipe(&dst);
    struct GuiPaint pat = GuiPaint::pattern(&src);
    pat.set_filter(GUI_FILTER_NEAREST);
    c.set_fill_paint(&pat);
    c.fill_rect(0.0f, 0.0f, 20.0f, 20.0f);
    ASSERT_EQ(dst.get_pixel(8, 8), 0xFF0000FFU);
    ASSERT_EQ(dst.get_pixel(9, 8), 0x0000FFFFU);
    ASSERT_EQ(dst.get_pixel(17, 13), 0xFF0000FFU);
    ASSERT_EQ(dst.get_pixel(18, 13), 0x0000FFFFU);
    c.free();
    // resampling
    struct GuiImage half = src.scaled(2, 2);
    ASSERT_TRUE(near(ch(&half, 0, 0, 0), 128, 1) && near(ch(&half, 0, 0, 2), 128, 1));
    half.free();
    struct GuiImage straight = GuiImage::new(1, 1);
    straight.set_pixel(0, 0, 0xFF000080U);
    straight.unpremultiply();
    ASSERT_EQ(straight.premultiplied, 0);
    straight.premultiply();
    ASSERT_TRUE(near(ch(&straight, 0, 0, 0), 255 * 128 / 255, 1));
    straight.free();
    // PNG round trip.
    ASSERT_TRUE(src.save_png("/tmp/safec_gui_canvas_test.png"));
    struct GuiImage back;
    ASSERT_TRUE(back.load_png_file("/tmp/safec_gui_canvas_test.png"));
    ASSERT_EQ(back.width, 4);
    ASSERT_EQ(back.get_pixel(1, 0), 0x0000FFFFU);
    back.free();
    src.free();
    dst.free();
}

void t_paths() {
    struct GuiPath p = GuiPath::new();
    ASSERT_TRUE(p.add_svg("M10 10 h 20 v 10 H 10 Z m 5 15 l 5 5 c 1 1 2 2 3 3 s 4 4 5 5 q 1 1 2 2 t 3 3 a 5 5 0 0 1 10 0"));
    struct GuiRect b = p.bounds();
    ASSERT_TRUE(b.x > 9.9f && b.y > 9.9f && b.w > 20.0f);
    ASSERT_FALSE(p.add_svg("M 10 10 L oops"));
    p.clear();
    ASSERT_TRUE(p.is_empty());
    p.arc(50.0f, 50.0f, 10.0f, 0.0f, 3.14159265f, 0);       // lower half circle (y down)
    b = p.bounds();
    ASSERT_TRUE(b.y > 49.9f && b.y + b.h > 59.0f && b.y + b.h < 61.5f);
    p.clear();
    p.move_to(0.0f, 0.0f);
    p.arc_to(10.0f, 0.0f, 10.0f, 10.0f, 5.0f);
    ASSERT_TRUE(p.curX > 9.9f && p.curX < 10.1f && p.curY > 4.9f && p.curY < 5.1f);
    p.free();
    struct GuiMatrix m = GuiMatrix::translation(5.0f, 7.0f).scale(2.0f, 4.0f).rotate(0.5f);
    struct GuiMatrix inv;
    ASSERT_TRUE(m.invert(&inv));
    struct GuiMatrix id = m.mul(inv);
    ASSERT_TRUE(id.a > 0.999f && id.a < 1.001f && id.e > -0.001f && id.e < 0.001f);
    float x = m.apply_x(3.0f, 4.0f);
    float y = m.apply_y(3.0f, 4.0f);
    float bx = inv.apply_x(x, y);
    ASSERT_TRUE(bx > 2.999f && bx < 3.001f);
    struct GuiMatrix sing = GuiMatrix::scaling(0.0f, 1.0f);
    ASSERT_FALSE(sing.invert(&inv));
}

// The golden scene: shapes, gradients, strokes, a clip and text.
static void draw_scene(&GuiCanvas c) {
    c.clear(GuiColor::hex(0xFFFFFFU));
    c.set_fill_color(GuiColor::hex(0x3366CCU));
    c.fill_rounded_rect(8.0f, 8.0f, 60.0f, 30.0f, 7.0f);
    struct GuiPaint g = GuiPaint::linear(76.0f, 0.0f, 152.0f, 0.0f);
    g.add_stop(0.0f, GuiColor::hex(0xFF0000U));
    g.add_stop(0.5f, GuiColor::hex(0xFFFF00U));
    g.add_stop(1.0f, GuiColor::hex(0x00AA00U));
    c.set_fill_paint(&g);
    c.fill_ellipse(114.0f, 23.0f, 38.0f, 15.0f);
    c.set_stroke_color(GuiColor::hex(0x000000U));
    c.set_line_width(3.0f);
    c.set_line_join(GUI_JOIN_ROUND);
    c.begin_path();
    c.move_to(10.0f, 60.0f); c.line_to(35.0f, 95.0f); c.quad_to(60.0f, 40.0f, 85.0f, 90.0f);
    c.cubic_to(100.0f, 50.0f, 120.0f, 120.0f, 150.0f, 55.0f);
    c.stroke();
    c.save();
    struct GuiPath clip = GuiPath::new();
    clip.circle(170.0f, 110.0f, 25.0f);
    c.clip_path(&clip);
    c.set_fill_color(GuiColor::rgba((unsigned char)200, (unsigned char)0, (unsigned char)100, (unsigned char)180));
    c.fill_rect(140.0f, 85.0f, 60.0f, 30.0f);
    c.restore();
    clip.free();
    c.set_fill_color(GuiColor::hex(0x202020U));
    c.set_font(GuiTypeface::fallback(), 15.0f);
    c.fill_text("SafeC Ag", 10.0f, 135.0f);
}

static int compare_golden(&GuiImage img, const &char path) {
    struct GuiImage ref;
    if (!ref.load_png_file(path)) { return -1; }
    if (ref.width != img.width || ref.height != img.height) { ref.free(); return -2; }
    struct GuiImage mine = img.copy();
    mine.unpremultiply();
    int bad = 0;
    int y = 0;
    while (y < img.height) {
        int x = 0;
        while (x < img.width) {
            int k = 0;
            while (k < 4) {
                if (!near(ch(&mine, x, y, k), ch(&ref, x, y, k), 3)) { bad = bad + 1; }
                k = k + 1;
            }
            x = x + 1;
        }
        y = y + 1;
    }
    mine.free();
    ref.free();
    return bad;
}

void t_golden() {
    struct GuiImage img = GuiImage::new(200, 150);
    struct GuiCanvas c = GuiCanvas::for_image(&img);
    draw_scene(&c);
    c.free();
    int bad = compare_golden(&img, "fixtures/gui/canvas_scene.png");
    if (bad != 0) { img.save_png("/tmp/safec_gui_canvas_scene_actual.png"); }
    ASSERT_EQ(bad, 0);
    img.free();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("rects: exact and fractional coverage", t_rects);
        t.add("circle area and anti-aliasing", t_coverage_circle);
        t.add("fill rules", t_fill_rules);
        t.add("strokes: width, caps, joins, dashes", t_strokes);
        t.add("gradients and paint alpha", t_gradients);
        t.add("transforms, save/restore, clipping", t_transform_state_clip);
        t.add("blend modes", t_blend);
        t.add("images, patterns, resampling, PNG", t_images);
        t.add("paths and matrices", t_paths);
        t.add("golden scene", t_golden);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
