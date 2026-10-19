// std::serial — value, json, csv, xml, yaml, html, protobuf, markdown
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/serial/csv.h>
#include <std/serial/xml.h>
#include <std/serial/yaml.h>
#include <std/serial/html.h>
#include <std/serial/protobuf.h>
#include <std/serial/markdown.h>

static struct Value sample() {
    struct Value o = value_object();
    value_object_set(&o, "name", value_string("Ada \"L\""));
    value_object_set(&o, "age", value_int(36LL));
    value_object_set(&o, "pi", value_float(3.5));
    value_object_set(&o, "ok", value_bool(1));
    value_object_set(&o, "nil", value_null());
    struct Value arr = value_array();
    value_array_push(&arr, value_int(1LL));
    value_array_push(&arr, value_int(2LL));
    value_array_push(&arr, value_string("three"));
    value_object_set(&o, "list", arr);
    return o;
}

void t_value() {
    struct Value o = sample();
    ASSERT_EQ(o.object_len(), 6);
    ?&Value age = o.object_get("age");
    ASSERT_TRUE(age != null);
    unsafe {
        ASSERT_EQ(age->as_int(), 36);
        ?&Value missing = o.object_get("nope");
        ASSERT_TRUE(missing == null);
    }
    struct Value c = value_clone(&o);
    ASSERT_EQ(c.object_len(), 6);
    value_free(&c);
    value_free(&o);
}

void t_json() {
    unsafe {
        struct Value o = sample();
        struct String js = value_to_json(&o);
        int ok = 0;
        struct Value back = json_parse(js.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        struct String js2 = value_to_json(&back);
        ASSERT_STR_EQ(js.as_ptr(), js2.as_ptr());
        ?&Value nm = back.object_get("name");
        ASSERT_STR_EQ(nm->as_string(), "Ada \"L\"");
        ?&Value lst = back.object_get("list");
        ASSERT_EQ(lst->array_len(), 3);
        struct Value nested = json_parse("{\"a\":[1,{\"b\":[true,false,null]},-2.5e3],\"u\":\"\\u00e9\\n\"}", &ok);
        ASSERT_TRUE(ok);
        ?&Value a = nested.object_get("a");
        ASSERT_EQ(a->array_len(), 3);
        ASSERT_TRUE(a->array_at(2UL).as_float() == -2500.0);
        ?&Value u = nested.object_get("u");
        ASSERT_STR_EQ(u->as_string(), "é\n");
        struct Value bad = json_parse("{\"a\":", &ok);
        ASSERT_FALSE(ok);
        value_free(&bad);
        value_free(&nested);
        js.free(); js2.free();
        value_free(&back);
        value_free(&o);
    }
}

void t_csv() {
    unsafe {
        int ok = 0;
        struct Value rows = csv_parse("a,b,c\n1,\"x,y\",3\n4,5,\"q\"\"q\"\n", &ok);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(rows.array_len() >= 2UL);
        struct String out = value_to_csv(&rows);
        struct Value rows2 = csv_parse(out.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(rows2.array_len(), rows.array_len());
        struct String out2 = value_to_csv(&rows2);
        ASSERT_STR_EQ(out.as_ptr(), out2.as_ptr());
        out.free(); out2.free();
        value_free(&rows); value_free(&rows2);
    }
}

void t_xml_yaml_html() {
    unsafe {
        struct Value o = sample();
        int ok = 0;
        struct String x = value_to_xml(&o, "root");
        struct Value xb = xml_parse(x.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        struct String x2 = value_to_xml(&xb, "root");
        ASSERT_STR_EQ(x.as_ptr(), x2.as_ptr());
        struct String y = value_to_yaml(&o);
        struct Value yb = yaml_parse(y.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        struct String y2 = value_to_yaml(&yb);
        ASSERT_STR_EQ(y.as_ptr(), y2.as_ptr());
        struct String h = value_to_html(&o);
        struct Value hb = html_parse(h.as_ptr(), &ok);
        ASSERT_TRUE(ok);
        struct String h2 = value_to_html(&hb);
        ASSERT_STR_EQ(h.as_ptr(), h2.as_ptr());
        struct String esc = string_new();
        html_escape(&esc, "<a href=\"x\">&</a>");
        ASSERT_TRUE(esc.contains("&lt;a"));
        ASSERT_TRUE(esc.contains("&amp;"));
        esc.free();
        x.free(); x2.free(); y.free(); y2.free(); h.free(); h2.free();
        value_free(&xb); value_free(&yb); value_free(&hb); value_free(&o);
    }
}

void t_protobuf() {
    unsafe {
        struct String buf = string_new();
        pb_write_int64_field(&buf, 1, 150LL);
        // Canonical example from the protobuf encoding guide: field 1 = 150 → 08 96 01
        ASSERT_EQ(buf.length(), 3);
        ASSERT_EQ((unsigned char)buf.char_at(0UL), 0x08);
        ASSERT_EQ((unsigned char)buf.char_at(1UL), 0x96);
        ASSERT_EQ((unsigned char)buf.char_at(2UL), 0x01);
        pb_write_sint64_field(&buf, 2, -3LL);
        pb_write_string_field(&buf, 3, "hi");
        pb_write_double_field(&buf, 4, 2.25);
        pb_write_bool_field(&buf, 5, 1);
        struct PbReader r = pb_reader_new((const unsigned char*)buf.as_ptr(), buf.length());
        int fn_ = 0;
        int wt = 0;
        long long iv = 0LL;
        ASSERT_TRUE(pb_read_tag(&r, &fn_, &wt));
        ASSERT_EQ(fn_, 1);
        pb_read_int64(&r, &iv);
        ASSERT_EQ(iv, 150);
        pb_read_tag(&r, &fn_, &wt);
        pb_read_sint64(&r, &iv);
        ASSERT_EQ(iv, -3);
        pb_read_tag(&r, &fn_, &wt);
        ASSERT_EQ(wt, PB_WIRE_LEN);
        const unsigned char* sd = (const unsigned char*)0;   // out-param typed by pb_read_length_delimited's API
        unsigned long sl = 0UL;
        pb_read_length_delimited(&r, &sd, &sl);
        ASSERT_EQ(sl, 2);
        ASSERT_EQ(sd[0], 'h');
        pb_read_tag(&r, &fn_, &wt);
        double dv = 0.0;
        pb_read_double(&r, &dv);
        ASSERT_TRUE(dv == 2.25);
        pb_read_tag(&r, &fn_, &wt);
        ASSERT_TRUE(pb_skip_field(&r, wt));
        ASSERT_FALSE(pb_reader_has_more(&r));
        buf.free();
    }
}

void t_markdown() {
    unsafe {
        &heap MdNode doc = md_parse("# Title\n\nSome **bold** and *it* with `code` and [link](http://x).\n\n- a\n- b\n\n```\nfn()\n```\n");
        struct String html = md_render_html(doc);
        ASSERT_TRUE(html.contains("<h1>Title</h1>"));
        ASSERT_TRUE(html.contains("<strong>bold</strong>"));
        ASSERT_TRUE(html.contains("<em>it</em>"));
        ASSERT_TRUE(html.contains("<code>code</code>"));
        ASSERT_TRUE(html.contains("<a href=\"http://x\">link</a>"));
        ASSERT_TRUE(html.contains("<li>"));
        struct String md = md_render_markdown(doc);
        &heap MdNode doc2 = md_parse(md.as_ptr());
        struct String html2 = md_render_html(doc2);
        ASSERT_STR_EQ(html.as_ptr(), html2.as_ptr());
        html.free(); html2.free(); md.free();
        md_free(doc); md_free(doc2);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("value", t_value);
        t.add("json", t_json);
        t.add("csv", t_csv);
        t.add("xml/yaml/html", t_xml_yaml_html);
        t.add("protobuf", t_protobuf);
        t.add("markdown", t_markdown);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
