// safec-doc — link check over a generated reference (see doctool.h).
//
// Every relative href= / src= in every .html under the directory must name
// a file there, and a '#fragment' an id="" on the page it names. Absolute
// URLs (https:, mailto:, ...) are not fetched.
#include "doctool.h"
#include <std/io.h>
#include <std/io_file.h>
#include <std/dir.h>
#include <std/path.h>

// Index of 'needle' in 'hay' at or after 'from', or -1.
static long long find_from(const &String hay, const &String needle, unsigned long from) {
    if (needle.len == 0UL || hay.len < needle.len) { return -1LL; }
    long long r = -1LL;
    unsafe {
        const char* h = (const char*)hay.as_ptr();
        const char* n = (const char*)needle.as_ptr();
        unsigned long last = hay.len - needle.len;
        for (unsigned long i = from; i <= last; i = i + 1UL) {
            if (h[i] != n[0]) { continue; }
            unsigned long j = 1UL;
            while (j < needle.len && h[i + j] == n[j]) { j = j + 1UL; }
            if (j == needle.len) { r = (long long)i; break; }
        }
    }
    return r;
}

// Appends every attribute value 'attr="…"' in 'html' to 'out'.
static void collect_attr(const &String html, const &char attr, &Vec<struct String> out) {
    struct String needle = String::from(" ");
    needle.push(attr);
    needle.push("=\"");
    unsigned long k = 0UL;
    for (;;) {
        long long at = find_from(html, &needle, k);
        if (at < 0LL) { break; }
        unsigned long s = (unsigned long)at + needle.len;
        unsigned long e = s;
        while (e < html.len && html.char_at(e) != (int)'"') { e = e + 1UL; }
        out.push(html.substr(s, e));
        k = e;
    }
    needle.free();
}

// "&amp;" -> "&" (the only entity attribute values here carry).
static struct String unescape_attr(const &String v) {
    struct String r = v.clone();
    r.replace_all("&amp;", "&");
    r.replace_all("&quot;", "\"");
    r.replace_all("&#39;", "'");
    return r;
}

// "https://…", "mailto:…", "//host/…": any URL with a scheme.
static int is_external(const &String v) {
    if (v.starts_with("//")) { return 1; }
    for (unsigned long i = 0UL; i < v.len; i = i + 1UL) {
        int ch = v.char_at(i);
        if (ch == (int)':') { return i > 0UL; }
        int ok = (ch >= (int)'a' && ch <= (int)'z') || (ch >= (int)'A' && ch <= (int)'Z') ||
                 (i > 0UL && ((ch >= (int)'0' && ch <= (int)'9') || ch == (int)'+' || ch == (int)'.' || ch == (int)'-'));
        if (!ok) { return 0; }
    }
    return 0;
}

int check_links(const &char dir) {
    struct Vec<struct String> files = Vec::new();
    if (!dir_walk(dir, &files)) {
        eprint("safec-doc: cannot read ");
        eprintln(dir);
        files.drop();
        return 1;
    }
    // Every "<page>#<id>" and every file.
    struct StrMap<int> ids = StrMap::new();
    struct StrMap<int> exists = StrMap::new();
    struct Vec<struct String> pages = Vec::new();
    for (unsigned long i = 0UL; i < files.len; i = i + 1UL) {
        &String rel = files.at(i);
        exists.insert(rel.as_ptr(), 1);
        if (!rel.ends_with(".html")) { continue; }
        struct String full = path_join(dir, rel.as_ptr());
        struct String html = String::new();
        File::read_to_string(full.as_ptr(), &html);
        struct Vec<struct String> found = Vec::new();
        collect_attr(&html, "id", &found);
        for (unsigned long k = 0UL; k < found.len; k = k + 1UL) {
            struct String key = rel.clone();
            key.push("#");
            struct String v = unescape_attr(found.at(k));
            key.push_str(&v);
            ids.insert(key.as_ptr(), 1);
            key.free(); v.free();
        }
        found.free();
        pages.push(rel.clone());
        full.free(); html.free();
    }
    int bad = 0;
    unsigned long checked = 0UL;
    for (unsigned long i = 0UL; i < pages.len; i = i + 1UL) {
        &String rel = pages.at(i);
        struct String full = path_join(dir, rel.as_ptr());
        struct String html = String::new();
        File::read_to_string(full.as_ptr(), &html);
        struct Vec<struct String> refs = Vec::new();
        collect_attr(&html, "href", &refs);
        collect_attr(&html, "src", &refs);
        struct String page_dir = path_dirname(rel.as_ptr());
        for (unsigned long k = 0UL; k < refs.len; k = k + 1UL) {
            struct String v = unescape_attr(refs.at(k));
            if (v.len == 0UL || is_external(&v)) { v.free(); continue; }
            checked = checked + 1UL;
            struct String path = v.clone();
            struct String frag = String::new();
            long long hash = v.find_char('#');
            if (hash >= 0LL) {
                std::replace(&frag, v.substr((unsigned long)hash + 1UL, v.len)).drop();
                path.truncate((unsigned long)hash);
            }
            struct String target = String::new();
            if (path.len == 0UL) {
                target.push_str(rel);
            } else {
                struct String joined = page_dir.len > 0UL && !page_dir.eq_cstr(".") ? path_join(page_dir.as_ptr(), path.as_ptr()) : path.clone();
                std::replace(&target, path_normalize(joined.as_ptr())).drop();
                joined.free();
                if (target.ends_with("/") || target.eq_cstr(".")) { target.push("index.html"); }
                struct String with_index = target.clone();
                with_index.push("/index.html");
                if (!exists.contains(target.as_ptr()) && exists.contains(with_index.as_ptr())) {
                    target.free();
                    target = with_index.clone();
                }
                with_index.free();
            }
            // Outside the reference (a link back to the hosting site): not ours.
            if (target.starts_with("../") || target.eq_cstr("..")) {
                v.free(); path.free(); frag.free(); target.free();
                continue;
            }
            int ok = exists.contains(target.as_ptr());
            if (ok && frag.len > 0UL) {
                struct String key = target.clone();
                key.push("#");
                key.push_str(&frag);
                ok = ids.contains(key.as_ptr());
                key.free();
            }
            if (!ok) {
                eprint(frag.len > 0UL && exists.contains(target.as_ptr()) ? "dead anchor: " : "dead link: ");
                eprint(rel.as_ptr());
                eprint(" -> ");
                eprintln(v.as_ptr());
                bad = bad + 1;
            }
            v.free(); path.free(); frag.free(); target.free();
        }
        refs.free();
        page_dir.free(); full.free(); html.free();
    }
    print("safec-doc: checked ");
    print_int((long long)checked);
    print(" links in ");
    print_int((long long)pages.len);
    print(" pages: ");
    print_int((long long)bad);
    println(" broken");
    files.free(); pages.free(); ids.free(); exists.free();
    return bad;
}
