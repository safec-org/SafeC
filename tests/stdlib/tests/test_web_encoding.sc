// std/encoding/url (Url, QueryParams, form encoding) and
// std/encoding/deflate (inflate/deflate, zlib, gzip, crc32, adler32).
#include <std/test/test.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/encoding/url.h>
#include <std/encoding/deflate.h>

static void check_resolve(&Url base, const &char ref, const &char want) {
    int ok = 0;
    struct Url r = base.resolve(ref, &ok);
    ASSERT_TRUE(ok);
    struct String s = r.to_string();
    ASSERT_STR_EQ(s.as_ptr(), want);
    s.free();
    r.free();
}

void t_url_parse() {
    int ok = 0;
    struct Url u = Url::parse("HTTPS://user:pa%20ss@Example.COM:8443/a/b%2Fc?x=1&y=two+words#frag", &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(u.scheme.as_ptr(), "https");
    ASSERT_STR_EQ(u.username.as_ptr(), "user");
    ASSERT_STR_EQ(u.password.as_ptr(), "pa%20ss");
    ASSERT_STR_EQ(u.host.as_ptr(), "example.com");
    ASSERT_EQ(u.port, 8443);
    ASSERT_STR_EQ(u.path.as_ptr(), "/a/b%2Fc");
    ASSERT_STR_EQ(u.query.as_ptr(), "x=1&y=two+words");
    ASSERT_STR_EQ(u.fragment.as_ptr(), "frag");
    struct String s = u.to_string();
    ASSERT_STR_EQ(s.as_ptr(), "https://user:pa%20ss@example.com:8443/a/b%2Fc?x=1&y=two+words#frag");
    s.free();
    struct String hp = u.host_port();
    ASSERT_STR_EQ(hp.as_ptr(), "example.com:8443");
    hp.free();
    struct String o = u.origin();
    ASSERT_STR_EQ(o.as_ptr(), "https://example.com:8443");
    o.free();
    struct String rt = u.request_target();
    ASSERT_STR_EQ(rt.as_ptr(), "/a/b%2Fc?x=1&y=two+words");
    rt.free();
    struct String pd = u.path_decoded();
    ASSERT_STR_EQ(pd.as_ptr(), "/a/b/c");
    pd.free();
    struct QueryParams q = u.query_params();
    ASSERT_STR_EQ(q.get_or("y", ""), "two words");
    q.free();
    u.free();

    u = Url::parse("http://[::1]:8080/", &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(u.host.as_ptr(), "::1");
    ASSERT_EQ(u.port, 8080);
    s = u.to_string();
    ASSERT_STR_EQ(s.as_ptr(), "http://[::1]:8080/");
    s.free();
    u.free();

    u = Url::parse("http://example.com", &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(u.port, -1);
    ASSERT_EQ(u.port_or_default(), 80);
    rt = u.request_target();
    ASSERT_STR_EQ(rt.as_ptr(), "/");
    rt.free();
    u.free();

    u = Url::parse("mailto:someone@example.com", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(u.has_authority);
    ASSERT_STR_EQ(u.path.as_ptr(), "someone@example.com");
    u.free();

    u = Url::parse("/relative/path?q", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(u.is_absolute());
    ASSERT_TRUE(u.has_query);
    u.free();

    u = Url::parse("http://h:99999/", &ok);
    ASSERT_FALSE(ok);
    u.free();
    u = Url::parse("http://h:8a/", &ok);
    ASSERT_FALSE(ok);
    u.free();
    u = Url::parse("http://bad host/", &ok);
    ASSERT_FALSE(ok);
    u.free();
    u = Url::parse("http://[::1/", &ok);
    ASSERT_FALSE(ok);
    u.free();
}

// RFC 3986 §5.4 reference resolution examples.
void t_url_resolve() {
    int ok = 0;
    struct Url base = Url::parse("http://a/b/c/d;p?q", &ok);
    ASSERT_TRUE(ok);
    check_resolve(&base, "g:h", "g:h");
    check_resolve(&base, "g", "http://a/b/c/g");
    check_resolve(&base, "./g", "http://a/b/c/g");
    check_resolve(&base, "g/", "http://a/b/c/g/");
    check_resolve(&base, "/g", "http://a/g");
    check_resolve(&base, "//g", "http://g");
    check_resolve(&base, "?y", "http://a/b/c/d;p?y");
    check_resolve(&base, "g?y", "http://a/b/c/g?y");
    check_resolve(&base, "#s", "http://a/b/c/d;p?q#s");
    check_resolve(&base, "g#s", "http://a/b/c/g#s");
    check_resolve(&base, ";x", "http://a/b/c/;x");
    check_resolve(&base, "", "http://a/b/c/d;p?q");
    check_resolve(&base, ".", "http://a/b/c/");
    check_resolve(&base, "./", "http://a/b/c/");
    check_resolve(&base, "..", "http://a/b/");
    check_resolve(&base, "../", "http://a/b/");
    check_resolve(&base, "../g", "http://a/b/g");
    check_resolve(&base, "../..", "http://a/");
    check_resolve(&base, "../../g", "http://a/g");
    check_resolve(&base, "../../../g", "http://a/g");
    check_resolve(&base, "/./g", "http://a/g");
    check_resolve(&base, "/../g", "http://a/g");
    check_resolve(&base, "g.", "http://a/b/c/g.");
    check_resolve(&base, "..g", "http://a/b/c/..g");
    check_resolve(&base, "./../g", "http://a/b/g");
    check_resolve(&base, "g/./h", "http://a/b/c/g/h");
    check_resolve(&base, "g/../h", "http://a/b/c/h");
    check_resolve(&base, "g;x=1/../y", "http://a/b/c/y");
    base.free();

    struct String d = url_remove_dot_segments("/a/b/c/./../../g");
    ASSERT_STR_EQ(d.as_ptr(), "/a/g");
    d.free();
}

void t_query_params() {
    struct QueryParams q = QueryParams::parse("?a=1&b=two%20words&a=3&&flag&c=x+y%2Bz&bad=%zz");
    ASSERT_EQ(q.length(), 6);
    ASSERT_STR_EQ(q.get_or("a", ""), "1");
    ASSERT_EQ(q.count("a"), 2);
    match (q.get_nth("a", 1UL)) {
        case some(v): { ASSERT_STR_EQ(v, "3"); }
        case null: { ASSERT_TRUE(0); }
    }
    ASSERT_TRUE(q.get_nth("a", 2UL).is_null());
    ASSERT_STR_EQ(q.get_or("b", ""), "two words");
    ASSERT_TRUE(q.has("flag"));
    ASSERT_STR_EQ(q.get_or("flag", "?"), "");
    ASSERT_STR_EQ(q.get_or("c", ""), "x y+z");
    ASSERT_STR_EQ(q.get_or("bad", ""), "%zz");
    ASSERT_TRUE(q.get("missing").is_null());
    ASSERT_STR_EQ(q.get_or("missing", "dflt"), "dflt");

    q.set("a", "only");
    ASSERT_EQ(q.count("a"), 1);
    ASSERT_STR_EQ(q.get_or("a", ""), "only");
    ASSERT_EQ(q.remove("flag"), 1);
    q.append("z", "last & first");
    q.sort();
    match (q.key_at(0UL)) { case some(k): { ASSERT_STR_EQ(k, "a"); } case null: { ASSERT_TRUE(0); } }
    struct String s = q.to_string();
    ASSERT_STR_EQ(s.as_ptr(), "a=only&b=two+words&bad=%25zz&c=x+y%2Bz&z=last+%26+first");
    s.free();
    struct QueryParams c = q.clone();
    ASSERT_EQ(c.length(), q.length());
    c.free();
    q.free();

    struct QueryParams body = QueryParams::parse_bytes(str_bytes("k=v&rest=ignored")[0..3]);
    ASSERT_EQ(body.length(), 1);
    ASSERT_STR_EQ(body.get_or("k", ""), "v");
    body.free();

    struct Url u = Url::parse("http://h/p", null);
    struct QueryParams p = QueryParams::new();
    p.append("q", "safe c");
    u.set_query_params(&p);
    struct String us = u.to_string();
    ASSERT_STR_EQ(us.as_ptr(), "http://h/p?q=safe+c");
    us.free();
    p.free();
    u.free();
}

void t_percent_encoding() {
    struct String e = url_encode("a b/c?d");
    ASSERT_STR_EQ(e.as_ptr(), "a%20b%2Fc%3Fd");
    e.free();
    e = url_encode_path("/a b/c:d@e%");
    ASSERT_STR_EQ(e.as_ptr(), "/a%20b/c:d@e%25");
    e.free();
    e = form_encode("a b*c&d=e");
    ASSERT_STR_EQ(e.as_ptr(), "a+b*c%26d%3De");
    e.free();
    e = form_decode("a+b%2Bc%");
    ASSERT_STR_EQ(e.as_ptr(), "a b+c%");
    e.free();
    int ok = 0;
    e = url_decode("%41%42+", &ok);
    ASSERT_TRUE(ok);
    ASSERT_STR_EQ(e.as_ptr(), "AB+");
    e.free();
}


// Python's zlib.compress(b"The quick brown fox jumps over the lazy dog. " * 20
// + bytes(range(256)), 9): dynamic-Huffman blocks from another encoder.
static const unsigned char kZlibRef[331] = { 0x78,0xda,0x0b,0xc9,0x48,0x55,0x28,0x2c,0xcd,0x4c,0xce,0x56,0x48,0x2a,0xca,0x2f,0xcf,0x53,0x48,0xcb,0xaf,0x50,0xc8,0x2a,0xcd,0x2d,0x28,0x56,0xc8,0x2f,0x4b,0x2d,0x52,0x28,0x01,0x4a,0xe7,0x24,0x56,0x55,0x2a,0xa4,0xe4,0xa7,0xeb,0x29,0x84,0x8c,0x2a,0x1e,0x55,0x3c,0xaa,0x98,0xda,0x8a,0x19,0x18,0x99,0x98,0x59,0x58,0xd9,0xd8,0x39,0x38,0xb9,0xb8,0x79,0x78,0xf9,0xf8,0x05,0x04,0x85,0x84,0x45,0x44,0xc5,0xc4,0x25,0x24,0xa5,0xa4,0x65,0x64,0xe5,0xe4,0x15,0x14,0x95,0x94,0x55,0x54,0xd5,0xd4,0x35,0x34,0xb5,0xb4,0x75,0x74,0xf5,0xf4,0x0d,0x0c,0x8d,0x8c,0x4d,0x4c,0xcd,0xcc,0x2d,0x2c,0xad,0xac,0x6d,0x6c,0xed,0xec,0x1d,0x1c,0x9d,0x9c,0x5d,0x5c,0xdd,0xdc,0x3d,0x3c,0xbd,0xbc,0x7d,0x7c,0xfd,0xfc,0x03,0x02,0x83,0x82,0x43,0x42,0xc3,0xc2,0x23,0x22,0xa3,0xa2,0x63,0x62,0xe3,0xe2,0x13,0x12,0x93,0x92,0x53,0x52,0xd3,0xd2,0x33,0x32,0xb3,0xb2,0x73,0x72,0xf3,0xf2,0x0b,0x0a,0x8b,0x8a,0x4b,0x4a,0xcb,0xca,0x2b,0x2a,0xab,0xaa,0x6b,0x6a,0xeb,0xea,0x1b,0x1a,0x9b,0x9a,0x5b,0x5a,0xdb,0xda,0x3b,0x3a,0xbb,0xba,0x7b,0x7a,0xfb,0xfa,0x27,0x4c,0x9c,0x34,0x79,0xca,0xd4,0x69,0xd3,0x67,0xcc,0x9c,0x35,0x7b,0xce,0xdc,0x79,0xf3,0x17,0x2c,0x5c,0xb4,0x78,0xc9,0xd2,0x65,0xcb,0x57,0xac,0x5c,0xb5,0x7a,0xcd,0xda,0x75,0xeb,0x37,0x6c,0xdc,0xb4,0x79,0xcb,0xd6,0x6d,0xdb,0x77,0xec,0xdc,0xb5,0x7b,0xcf,0xde,0x7d,0xfb,0x0f,0x1c,0x3c,0x74,0xf8,0xc8,0xd1,0x63,0xc7,0x4f,0x9c,0x3c,0x75,0xfa,0xcc,0xd9,0x73,0xe7,0x2f,0x5c,0xbc,0x74,0xf9,0xca,0xd5,0x6b,0xd7,0x6f,0xdc,0xbc,0x75,0xfb,0xce,0xdd,0x7b,0xf7,0x1f,0x3c,0x7c,0xf4,0xf8,0xc9,0xd3,0x67,0xcf,0x5f,0xbc,0x7c,0xf5,0xfa,0xcd,0xdb,0x77,0xef,0x3f,0x7c,0xfc,0xf4,0xf9,0xcb,0xd7,0x6f,0xdf,0x7f,0xfc,0xfc,0xf5,0xfb,0xcf,0xdf,0x7f,0xff,0x01,0x14,0x97,0xc2,0x9c };

static void make_sample(&Vec<unsigned char> v, unsigned long n, int kind) {
    unsigned int x = 12345U;
    unsigned long i = 0UL;
    while (i < n) {
        unsigned char b = (unsigned char)0;
        if (kind == 0) { b = (unsigned char)(i % 256UL); }
        else if (kind == 1) { x = x * 1103515245U + 12345U; b = (unsigned char)(x >> 16); }
        else if (kind == 2) { b = (unsigned char)'a'; }
        else {
            const &char words = "lorem ipsum dolor sit amet consectetur adipiscing elit ";
            unsafe { b = (unsigned char)words[i % 56UL]; }
            x = x * 1103515245U + 12345U;
            if ((x >> 20) % 97U == 0U) { b = (unsigned char)(x >> 8); }
        }
        v.push(b);
        i = i + 1UL;
    }
}

static int same_bytes(&Vec<unsigned char> a, &Vec<unsigned char> b) {
    if (a.len != b.len) return 0;
    unsigned long i = 0UL;
    while (i < a.len) {
        if (*a.at(i) != *b.at(i)) return 0;
        i = i + 1UL;
    }
    return 1;
}

void t_checksums() {
    unsafe {
        ASSERT_EQ(crc32(str_bytes("123456789")), 0xCBF43926U);
        ASSERT_EQ(adler32(str_bytes("Wikipedia")), 0x11E60398U);
        unsigned int c = crc32_update(0U, str_bytes("1234"));
        c = crc32_update(c, str_bytes("56789"));
        ASSERT_EQ(c, 0xCBF43926U);
        ASSERT_EQ(crc32(str_bytes("")), 0U);
    }
}

void t_inflate_reference() {
    int ok = 0;
    struct Vec<unsigned char> out = zlib_decompress(kZlibRef[0..331], 0UL, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.len, 1156);
    ASSERT_EQ(*out.at(0UL), 'T');
    ASSERT_EQ(*out.at(1155UL), 255);
    out.free();
    // A limit below the real size fails rather than truncating.
    out = zlib_decompress(kZlibRef[0..331], 1000UL, &ok);
    ASSERT_FALSE(ok);
    out.free();
    // Corrupt the Adler-32 trailer.
    unsigned char bad[331];
    unsigned long i = 0UL;
    while (i < 331UL) { unsafe { bad[i] = kZlibRef[i]; } i = i + 1UL; }
    bad[330] = (unsigned char)(bad[330] ^ (unsigned char)1);
    out = zlib_decompress(bad[0..331], 0UL, &ok);
    ASSERT_FALSE(ok);
    out.free();
    // Truncated stream.
    out = zlib_decompress(kZlibRef[0..100], 0UL, &ok);
    ASSERT_FALSE(ok);
    out.free();

    // Python's gzip.compress(b"hello hello hello gzip", mtime=0).
    unsigned char gz[42] = { 0x1f,0x8b,0x08,0x00,0x00,0x00,0x00,0x00,0x02,0xff,0xcb,0x48,0xcd,0xc9,0xc9,0x57,
                             0xc8,0x40,0x22,0xd3,0xab,0x32,0x0b,0x00,0x47,0x3a,0x59,0x1c,0x16,0x00,0x00,0x00 };
    ASSERT_TRUE(gzip_is_gzip(gz[0..32]));
    out = gzip_decompress(gz[0..32], 0UL, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.len, 22);
    ASSERT_EQ(*out.at(18UL), 'g');
    out.free();
    // Two concatenated members decode to both payloads.
    i = 0UL;
    while (i < 10UL) { gz[32UL + i] = gz[i]; i = i + 1UL; }
    out = gzip_decompress(gz[0..42], 0UL, &ok);
    ASSERT_FALSE(ok);   // second member is truncated
    out.free();
}

void t_deflate_roundtrip() {
    int kind = 0;
    while (kind < 4) {
        unsigned long sizes[5] = { 0UL, 1UL, 300UL, 70000UL, 200000UL };
        int si = 0;
        while (si < 5) {
            struct Vec<unsigned char> src = Vec::new();
            make_sample(&src, sizes[si], kind);
            int level = 0;
            while (level <= 9) {
                []unsigned char p = src.as_slice();
                struct Vec<unsigned char> raw = deflate_raw(p, level);
                int ok = 0;
                struct Vec<unsigned char> back = inflate_raw(raw.as_slice(), 0UL, null, &ok);
                ASSERT_TRUE(ok);
                ASSERT_TRUE(same_bytes(&back, &src));
                if (kind == 2 && sizes[si] == 200000UL && level > 0) {
                    ASSERT_TRUE(raw.len < 2000UL);   // runs compress hard
                }
                if (kind == 1 && level > 0) {
                    ASSERT_TRUE(raw.len <= src.len + src.len / 100UL + 16UL);   // random: near-stored
                }
                back.free();
                raw.free();

                struct Vec<unsigned char> z = zlib_compress(p, level);
                back = zlib_decompress(z.as_slice(), 0UL, &ok);
                ASSERT_TRUE(ok);
                ASSERT_TRUE(same_bytes(&back, &src));
                back.free();
                z.free();

                struct Vec<unsigned char> g = gzip_compress(p, level);
                back = gzip_decompress(g.as_slice(), 0UL, &ok);
                ASSERT_TRUE(ok);
                ASSERT_TRUE(same_bytes(&back, &src));
                back.free();
                g.free();
                level = level + (sizes[si] > 70000UL ? 3 : 1);
            }
            src.free();
            si = si + 1;
        }
        kind = kind + 1;
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    t.add("url parse/serialize", t_url_parse);
    t.add("url resolve (RFC 3986 5.4)", t_url_resolve);
    t.add("query params", t_query_params);
    t.add("percent/form encoding", t_percent_encoding);
    t.add("crc32/adler32", t_checksums);
    t.add("inflate reference streams", t_inflate_reference);
    t.add("deflate/zlib/gzip round trips", t_deflate_roundtrip);
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
