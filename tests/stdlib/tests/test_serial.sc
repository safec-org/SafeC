// std::serial — value, json, csv, xml, yaml, html, protobuf, markdown; std::scx
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
#include <std/scx/scx.h>

static struct Value sample() {
    struct Value o = value_object();
    o.object_set("name", value_string("Ada \"L\""));
    o.object_set("age", value_int(36LL));
    o.object_set("pi", value_float(3.5));
    o.object_set("ok", value_bool(1));
    o.object_set("nil", value_null());
    struct Value arr = value_array();
    arr.array_push(value_int(1LL));
    arr.array_push(value_int(2LL));
    arr.array_push(value_string("three"));
    o.object_set("list", arr);
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
    struct Value c = o.clone();
    ASSERT_EQ(c.object_len(), 6);
    c.free();
    o.free();
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
        bad.free();
        nested.free();
        js.free(); js2.free();
        back.free();
        o.free();
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
        rows.free(); rows2.free();
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
        struct String esc = String::new();
        html_escape(&esc, "<a href=\"x\">&</a>");
        ASSERT_TRUE(esc.contains("&lt;a"));
        ASSERT_TRUE(esc.contains("&amp;"));
        esc.free();
        x.free(); x2.free(); y.free(); y2.free(); h.free(); h2.free();
        xb.free(); yb.free(); hb.free(); o.free();
    }
}

void t_protobuf() {
    unsafe {
        struct String buf = String::new();
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
        ASSERT_TRUE(r.read_tag(&fn_, &wt));
        ASSERT_EQ(fn_, 1);
        r.read_int64(&iv);
        ASSERT_EQ(iv, 150);
        r.read_tag(&fn_, &wt);
        r.read_sint64(&iv);
        ASSERT_EQ(iv, -3);
        r.read_tag(&fn_, &wt);
        ASSERT_EQ(wt, PB_WIRE_LEN);
        unsigned char none = (unsigned char)0;
        &const unsigned char sd = &none;
        unsigned long sl = 0UL;
        r.read_length_delimited(&sd, &sl);
        ASSERT_EQ(sl, 2);
        ASSERT_EQ(sd[0], 'h');
        r.read_tag(&fn_, &wt);
        double dv = 0.0;
        r.read_double(&dv);
        ASSERT_TRUE(dv == 2.25);
        r.read_tag(&fn_, &wt);
        ASSERT_TRUE(r.skip_field(wt));
        ASSERT_FALSE(r.has_more());
        buf.free();
    }
}

void t_markdown() {
    unsafe {
        &heap MdNode doc = md_parse("# Title\n\nSome **bold** and *it* with `code` and [link](http://x).\n\n- a\n- b\n\n```\nfn()\n```\n");
        struct String html = doc.render_html();
        ASSERT_TRUE(html.contains("<h1>Title</h1>"));
        ASSERT_TRUE(html.contains("<strong>bold</strong>"));
        ASSERT_TRUE(html.contains("<em>it</em>"));
        ASSERT_TRUE(html.contains("<code>code</code>"));
        ASSERT_TRUE(html.contains("<a href=\"http://x\">link</a>"));
        ASSERT_TRUE(html.contains("<li>"));
        struct String md = doc.render_markdown();
        &heap MdNode doc2 = md_parse(md.as_ptr());
        struct String html2 = doc2.render_html();
        ASSERT_STR_EQ(html.as_ptr(), html2.as_ptr());
        html.free(); html2.free(); md.free();
        doc.free(); doc2.free();
    }
}

// The append-to-String writers equal the value_to_* convenience forms, and
// varints round-trip at their size boundaries.
void t_writers_varint() {
    unsafe {
        struct Value o = sample();
        ASSERT_EQ(o.kind_of(), VAL_OBJECT);
        ?&Value okv = o.object_get("ok");
        ASSERT_EQ(okv->as_bool(), 1);
        ASSERT_EQ(okv->kind_of(), VAL_BOOL);

        struct String a = String::from("prefix:");
        json_write(&o, &a);
        struct String j = value_to_json(&o);
        ASSERT_TRUE(a.starts_with("prefix:"));
        ASSERT_STR_EQ(&a.as_ptr()[7], j.as_ptr());
        a.clear();
        yaml_write(&o, &a);
        struct String y = value_to_yaml(&o);
        ASSERT_STR_EQ(a.as_ptr(), y.as_ptr());
        a.clear();
        xml_write(&o, "root", &a);
        struct String x = value_to_xml(&o, "root");
        ASSERT_STR_EQ(a.as_ptr(), x.as_ptr());
        a.clear();
        html_write(&o, &a);
        struct String h = value_to_html(&o);
        ASSERT_STR_EQ(a.as_ptr(), h.as_ptr());
        int ok = 0;
        struct Value rows = csv_parse("a,b\n1,2\n", &ok);
        a.clear();
        csv_write(&rows, &a);
        struct String c = value_to_csv(&rows);
        ASSERT_STR_EQ(a.as_ptr(), c.as_ptr());
        j.free(); y.free(); x.free(); h.free(); c.free(); a.free();
        rows.free();
        o.free();

        unsigned long long vals[6];
        vals[0] = 0ULL; vals[1] = 127ULL; vals[2] = 128ULL; vals[3] = 300ULL;
        vals[4] = 9223372036854775808ULL; vals[5] = 18446744073709551615ULL;
        struct String buf = String::new();
        int i = 0;
        while (i < 6) { pb_write_varint(&buf, vals[i]); i = i + 1; }
        ASSERT_EQ(buf.length(), 1 + 1 + 2 + 2 + 10 + 10);
        struct PbReader r = pb_reader_new((const unsigned char*)buf.as_ptr(), buf.length());
        int good = 1;
        i = 0;
        while (i < 6) {
            unsigned long long v = 0ULL;
            if (!r.read_varint(&v) || v != vals[i]) good = 0;
            i = i + 1;
        }
        ASSERT_TRUE(good);
        ASSERT_FALSE(r.has_more());
        unsigned long long v = 0ULL;
        ASSERT_FALSE(r.read_varint(&v));                      // nothing left
        struct PbReader trunc = pb_reader_new((const unsigned char*)"\x80\x80", 2UL);
        ASSERT_FALSE(trunc.read_varint(&v));                  // unterminated
        buf.free();

        struct String e = String::new();
        scx_append_esc(&e, "<a href='x'>\"&\"</a>");
        ASSERT_STR_EQ(e.as_ptr(), "&lt;a href=&#39;x&#39;&gt;&quot;&amp;&quot;&lt;/a&gt;");
        e.free();
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
        t.add("writers/varint/scx", t_writers_varint);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
