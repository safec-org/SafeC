// std::serial — value, json, csv, xml, yaml, html, protobuf, markdown; std::scx
#include <std/test/test.h>
#include <std/io.h>
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
    struct Value o = Value::object();
    o.object_set("name", Value::from_str("Ada \"L\""));
    o.object_set("age", Value::from_int(36LL));
    o.object_set("pi", Value::from_float(3.5));
    o.object_set("ok", Value::from_bool(1));
    o.object_set("nil", Value::null());
    struct Value arr = Value::array();
    arr.array_push(Value::from_int(1LL));
    arr.array_push(Value::from_int(2LL));
    arr.array_push(Value::from_str("three"));
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

// Any Format (value.h): encode, parse back, encode again — the two texts
// match. 1 on success.
generic<F: Format>
int round_trips(F f, const &Value v) {
    int ok = 0;
    struct String text = v.encode(f);
    struct Value back = Value::decode(f, text.as_ptr(), &ok);
    struct String again = back.encode(f);
    int same = 0;
    unsafe { same = str_eq(text.as_ptr(), again.as_ptr()); }
    int good = ok && same && text.length() > 0UL;
    text.free(); again.free(); back.free();
    return good;
}

void t_xml_yaml_html() {
    struct Value o = sample();
    ASSERT_TRUE(round_trips(Json::new(), &o));
    ASSERT_TRUE(round_trips(Yaml::new(), &o));
    ASSERT_TRUE(round_trips(Xml::new("root"), &o));
    ASSERT_TRUE(round_trips(Html::new(), &o));
    int ok = 0;
    struct Value rows = Value::decode(Csv::new(), "a,b\n1,\"x,y\"\n", &ok);   // CSV: rows of fields
    ASSERT_TRUE(ok);
    ASSERT_TRUE(round_trips(Csv::new(), &rows));
    struct String x = o.encode(Xml::new("doc"));
    ASSERT_TRUE(x.starts_with("<doc>"));
    struct Value bad = Value::decode(Json::new(), "[1,", &ok);
    ASSERT_FALSE(ok);
    struct String esc = String::new();
    html_escape(&esc, "<a href=\"x\">&</a>");
    ASSERT_TRUE(esc.contains("&lt;a"));
    ASSERT_TRUE(esc.contains("&amp;"));
    esc.free(); x.free();
    bad.free(); rows.free(); o.free();
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
        struct PbReader r = PbReader::new((const unsigned char*)buf.as_ptr(), buf.length());
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
        &heap MdNode doc = MdNode::parse("# Title\n\nSome **bold** and *it* with `code` and [link](http://x).\n\n- a\n- b\n\n```\nfn()\n```\n");
        struct String html = doc.render_html();
        ASSERT_TRUE(html.contains("<h1>Title</h1>"));
        ASSERT_TRUE(html.contains("<strong>bold</strong>"));
        ASSERT_TRUE(html.contains("<em>it</em>"));
        ASSERT_TRUE(html.contains("<code>code</code>"));
        ASSERT_TRUE(html.contains("<a href=\"http://x\">link</a>"));
        ASSERT_TRUE(html.contains("<li>"));
        struct String md = doc.render_markdown();
        &heap MdNode doc2 = MdNode::parse(md.as_ptr());
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
        struct PbReader r = PbReader::new((const unsigned char*)buf.as_ptr(), buf.length());
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
        struct PbReader trunc = PbReader::new((const unsigned char*)"\x80\x80", 2UL);
        ASSERT_FALSE(trunc.read_varint(&v));                  // unterminated
        buf.free();

        struct String e = String::new();
        scx_append_esc(&e, "<a href='x'>\"&\"</a>");
        ASSERT_STR_EQ(e.as_ptr(), "&lt;a href=&#39;x&#39;&gt;&quot;&amp;&quot;&lt;/a&gt;");
        e.free();
    }
}

// The object member 'k' (the object itself when missing — the asserts then fail).
static &Value fld(const &Value o, const &char k) {
    ?&Value r = o.object_get(k);
    match (r) { case some(x): { return x; } case null: { } }
    return o;
}

void t_yaml_flow_block() {
    unsafe {
        int ok = 0;
        struct Value v = yaml_parse("outline: [2, 4]\nempty: []\nmap: {a: 1, b: [x, \"y, z\"], c: {d: true}}\nurl: http://x.y/z\ntag: a#b # comment\nitems:\n  - [1, 2]\n  - {k: v}\n", &ok);
        ASSERT_TRUE(ok);
        &Value o = fld(&v, "outline");
        ASSERT_EQ(o.kind_of(), VAL_ARRAY);
        ASSERT_EQ(o.array_len(), 2);
        ASSERT_EQ(o.array_at(1UL).as_int(), 4);
        ASSERT_EQ(fld(&v, "empty").array_len(), 0);
        &Value m = fld(&v, "map");
        ASSERT_EQ(m.object_len(), 3);
        ASSERT_STR_EQ(m.object_key_at(1UL), "b");
        ASSERT_STR_EQ(m.object_value_at(1UL).array_at(1UL).as_string(), "y, z");
        ASSERT_TRUE(fld(fld(m, "c"), "d").as_bool());
        ASSERT_STR_EQ(fld(&v, "url").as_string(), "http://x.y/z");
        ASSERT_STR_EQ(fld(&v, "tag").as_string(), "a#b");
        &Value items = fld(&v, "items");
        ASSERT_EQ(items.array_at(0UL).array_len(), 2);
        ASSERT_STR_EQ(fld(items.array_at(1UL), "k").as_string(), "v");
        ASSERT_STR_EQ(v.object_key_at(0UL), "outline");
        v.free();

        struct Value b = yaml_parse("lit: |\n  line one\n    indented\n\n  after blank\nfold: >\n  a\n  b\n\n  c\nstrip: |-\n  x\nkeep: |+\n  y\n\nlist:\n  - |\n    in list\n  - plain\nnext: 1\n", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STR_EQ(fld(&b, "lit").as_string(), "line one\n  indented\n\nafter blank\n");
        ASSERT_STR_EQ(fld(&b, "fold").as_string(), "a b\nc\n");
        ASSERT_STR_EQ(fld(&b, "strip").as_string(), "x");
        ASSERT_STR_EQ(fld(&b, "keep").as_string(), "y\n\n");
        ASSERT_STR_EQ(fld(&b, "list").array_at(0UL).as_string(), "in list\n");
        ASSERT_STR_EQ(fld(&b, "list").array_at(1UL).as_string(), "plain");
        ASSERT_EQ(fld(&b, "next").as_int(), 1);
        b.free();

        struct Value bad = yaml_parse("x: [1, 2\n", &ok);
        ASSERT_TRUE(ok);   // an unclosed flow collection stays a plain string
        ASSERT_STR_EQ(fld(&bad, "x").as_string(), "[1, 2");
        bad.free();
    }
}

void t_html_text() {
    struct String u = String::new();
    html_unescape(&u, "a &lt;b&gt; &amp;amp; &#39;q&#x27; &mdash; &copy; &nbsp;|&bogus; & x &#0;");
    ASSERT_STR_EQ(u.as_ptr(), "a <b> &amp; 'q' \xe2\x80\x94 \xc2\xa9 \xc2\xa0|&bogus; & x &#0;");
    struct String t = String::new();
    html_strip_tags(&t, "<h2 id=\"x\">Title <a href=\"#x\">#</a></h2><p>One &amp; <code>two</code></p><script>if (a<b) {}</script><!-- c --><p title=\"a>b\">three</p>4 < 5");
    ASSERT_STR_EQ(t.as_ptr(), "Title # One &amp; two three 4 < 5");
    u.free(); t.free();
}

static struct String md_html(const &char src, int ext) {
    struct MdOptions o = MdOptions::commonmark();
    if (ext) { o = MdOptions::gfm(); o.html = 1; o.containers = 1; o.heading_ids = 1; }
    &heap MdNode d = MdNode::parse_opts(src, &o);
    struct String h = d.render_html();
    d.free();
    return h;
}

static void md_is(const &char src, int ext, const &char want) {
    struct String h = md_html(src, ext);
    if (!h.eq_cstr(want)) { eprint("got: "); eprintln(h.as_ptr()); }
    ASSERT_STR_EQ(h.as_ptr(), want);
    h.free();
}

struct MdHookCtx { int links; };

static int md_test_hook(?&MdHookCtx ctx, const &MdNode n, int entering, &String out) {
    if (n.kind == MD_LINK && entering) {
        match (ctx) { case some(c): { c.links = c.links + 1; } case null: { } }
        out.push("<a class=\"x\" href=\"/base");
        out.push(n.text.as_ptr());
        out.push("\">");
        return 2;
    }
    if (n.kind == MD_CODE_BLOCK) {
        out.push("<pre class=\"hl\">");
        out.push(n.info.as_ptr());
        out.push("</pre>\n");
        return 1;
    }
    if (n.kind == MD_HEADING && !entering) {
        out.push(" <a href=\"#");
        out.push(n.id.as_ptr());
        out.push("\">#</a></h2>\n");
        return 1;
    }
    if (n.kind == MD_HEADING) {
        out.push("<h2 id=\"");
        out.push(n.id.as_ptr());
        out.push("\">");
        return 2;
    }
    return 0;
}

void t_markdown_ext() {
    // Inlines: escapes, code-span precedence, intraword '_', nesting in links.
    md_is("a\\*b\\* `x_y_z` snake_case_name *it* __b__ **a *b* c**\n", 0,
          "<p>a*b* <code>x_y_z</code> snake_case_name <em>it</em> <strong>b</strong> <strong>a <em>b</em> c</strong></p>\n");
    md_is("[`code` and *em*](/u \"T\") ![alt *x*](i.png) <https://a.b/c> <me@x.org>\n", 0,
          "<p><a href=\"/u\" title=\"T\"><code>code</code> and <em>em</em></a> <img src=\"i.png\" alt=\"alt x\"> <a href=\"https://a.b/c\">https://a.b/c</a> <a href=\"mailto:me@x.org\">me@x.org</a></p>\n");
    md_is("&lt;T&gt; &amp; &copy; &#65; a&b\nline  \nnext\\\nlast\n", 0,
          "<p>&lt;T&gt; &amp; \xc2\xa9 A a&amp;b\nline<br>\nnext<br>\nlast</p>\n");
    md_is("**a*\n", 0, "<p>*<em>a</em></p>\n");
    md_is("<b>raw</b> <!-- c -->\n", 0, "<p>&lt;b&gt;raw&lt;/b&gt; &lt;!-- c --&gt;</p>\n");
    md_is("<b>raw</b> x\n", 1, "<p><b>raw</b> x</p>\n");
    md_is("~~gone~~ see https://example.com/a_b. ok\n", 1,
          "<p><del>gone</del> see <a href=\"https://example.com/a_b\">https://example.com/a_b</a>. ok</p>\n");

    // Blocks: nested lists, code in items, loose lists, setext, indented code.
    md_is("- a\n  - b\n    - c\n- d\n\n  ```c\n  x\n  ```\n", 0,
          "<ul>\n<li>\n<p>a</p>\n<ul>\n<li>b\n<ul>\n<li>c</li>\n</ul>\n</li>\n</ul>\n</li>\n<li>\n<p>d</p>\n<pre><code class=\"language-c\">x\n</code></pre>\n</li>\n</ul>\n");
    md_is("3. x\n4. y\n", 0, "<ol start=\"3\">\n<li>x</li>\n<li>y</li>\n</ol>\n");
    md_is("Title\n=====\n\n    code\n\n> q *e*\nlazy\n\n---\n", 0,
          "<h1>Title</h1>\n<pre><code>code\n</code></pre>\n<blockquote>\n<p>q <em>e</em>\nlazy</p>\n</blockquote>\n<hr>\n");

    // Extensions: tables, containers, raw HTML blocks, heading ids.
    md_is("| a | `b\\|c` | d |\n|:--|:-:|--:|\n| *1* | 2 |\n", 1,
          "<table>\n<thead>\n<tr>\n<th style=\"text-align:left\">a</th>\n<th style=\"text-align:center\"><code>b|c</code></th>\n<th style=\"text-align:right\">d</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td style=\"text-align:left\"><em>1</em></td>\n<td style=\"text-align:center\">2</td>\n<td style=\"text-align:right\"></td>\n</tr>\n</tbody>\n</table>\n");
    md_is("::: warning `x` is *odd*\nbody\n::: details\ninner\n:::\n:::\n::: tip\nt\n:::\n", 1,
          "<div class=\"custom-block warning\">\n<p class=\"custom-block-title\"><code>x</code> is <em>odd</em></p>\n<p>body</p>\n<details class=\"custom-block details\"><summary>Details</summary>\n<p>inner</p>\n</details>\n</div>\n<div class=\"custom-block tip\"><p class=\"custom-block-title\">TIP</p>\n<p>t</p>\n</div>\n");
    md_is("<div class=\"x\">\n*not md*\n</div>\n\n<a id=\"anchor\"></a>\n\ntext\n", 1,
          "<div class=\"x\">\n*not md*\n</div>\n<p><a id=\"anchor\"></a></p>\n<p>text</p>\n");
    md_is("# A B\n## `__attribute__` {#__attribute__}\n## 2. Temporal Safety\n## A B\n## C/C++ & Rust?\n", 1,
          "<h1 id=\"a-b\">A B</h1>\n<h2 id=\"__attribute__\"><code>__attribute__</code></h2>\n<h2 id=\"_2-temporal-safety\">2. Temporal Safety</h2>\n<h2 id=\"a-b-1\">A B</h2>\n<h2 id=\"c-c-rust\">C/C++ &amp; Rust?</h2>\n");

    // Slugs, plain text.
    struct String slug = String::new();
    md_slugify("Hello, World! \xe2\x80\x9cq\xe2\x80\x9d \xed\x95\x9c", &slug);
    ASSERT_STR_EQ(slug.as_ptr(), "hello-world-q-\xed\x95\x9c");
    slug.free();
    struct MdOptions o = MdOptions::gfm();
    o.heading_ids = 1;
    &heap MdNode d = MdNode::parse_opts("## *Hi* `there`\n\n```c\nint x;\n```\n[link](/a) and [x](/b)\n", &o);
    struct String pt = String::new();
    (*d.children.at(0UL)).plain_text(&pt);
    ASSERT_STR_EQ(pt.as_ptr(), "Hi there");
    pt.free();

    // Hooks.
    struct MdHookCtx ctx;
    ctx.links = 0;
    struct String hooked;
    unsafe { hooked = md_render_html_with(d, md_test_hook, (?&MdHookCtx)&ctx); }
    ASSERT_STR_EQ(hooked.as_ptr(), "<h2 id=\"hi-there\"><em>Hi</em> <code>there</code> <a href=\"#hi-there\">#</a></h2>\n<pre class=\"hl\">c</pre>\n<p><a class=\"x\" href=\"/base/a\">link</a> and <a class=\"x\" href=\"/base/b\">x</a></p>\n");
    ASSERT_EQ(ctx.links, 2);
    hooked.free();

    // Markdown round trip of the extended syntax.
    struct MdOptions all = MdOptions::gfm();
    all.containers = 1;
    &heap MdNode e = MdNode::parse_opts("::: tip Note\n- a\n  - b\n\n| x | y |\n|---|--:|\n| 1 | `2` |\n:::\n\n1. *x* ~~y~~\n", &all);
    struct String h1 = e.render_html();
    struct String back = e.render_markdown();
    &heap MdNode e2 = MdNode::parse_opts(back.as_ptr(), &all);
    struct String h2 = e2.render_html();
    ASSERT_STR_EQ(h1.as_ptr(), h2.as_ptr());
    h1.free(); h2.free(); back.free();
    e.free(); e2.free(); d.free();
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("value", t_value);
        t.add("json", t_json);
        t.add("csv", t_csv);
        t.add("xml/yaml/html", t_xml_yaml_html);
        t.add("protobuf", t_protobuf);
        t.add("markdown", t_markdown);
        t.add("markdown extensions/hooks", t_markdown_ext);
        t.add("yaml flow/block scalars, value order", t_yaml_flow_block);
        t.add("html unescape/strip_tags", t_html_text);
        t.add("writers/varint/scx", t_writers_varint);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
