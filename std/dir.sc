// SafeC Standard Library — Directories: the platform-independent tree
// operations (see dir.h). Dir, file_stat, dir_create, dir_remove and
// file_remove come from dir_posix.sc / dir_win32.sc.
#pragma once
#include <std/dir.h>
#include <std/path.h>
#include <std/io_file.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

namespace std {

struct DirEntry DirEntry::new() {
    struct DirEntry e;
    e.name = String::new();
    e.kind = DIR_NONE;
    return e;
}

void DirEntry::free() { self.drop(); }

void DirEntry::drop() {
    self.name.free();
}

int path_kind(const &char path) {
    struct FileStat st;
    file_stat(path, &st);
    return st.kind;
}

int path_exists(const &char path) {
    return path_kind(path) != DIR_NONE ? 1 : 0;
}

int dir_create_all(const &char path) {
    int kind = path_kind(path);
    if (kind == DIR_DIR) { return 1; }
    if (kind != DIR_NONE) { return 0; }   // a file is in the way
    struct String parent = path_dirname(path);
    int ok = 1;
    if (parent.len > 0UL && !parent.eq_cstr(path)) { ok = dir_create_all(parent.as_ptr()); }
    parent.free();
    if (!ok) { return 0; }
    return dir_create(path);
}

int dir_remove_all(const &char path) {
    int kind = path_kind(path);
    if (kind == DIR_NONE) { return 1; }
    if (kind != DIR_DIR) {
        if (file_remove(path)) { return 1; }
        return dir_remove(path);   // a Windows directory link
    }
    int ok = 1;
    ?&Dir d = Dir::open(path);
    match (d) {
        case some(h): {
            struct DirEntry e = DirEntry::new();
            while (h.next(&e)) {
                struct String child = path_join(path, e.name.as_ptr());
                if (!dir_remove_all(child.as_ptr())) { ok = 0; }
                child.free();
            }
            e.free();
            h.close();
        }
        case null: { ok = 0; }
    }
    if (!dir_remove(path)) { ok = 0; }
    return ok;
}

int file_copy(const &char from, const &char to) {
    ?&File src = File::open(from, "rb");
    match (src) {
        case some(in): {
            ?&File dst = File::open(to, "wb");
            int ok = 1;
            match (dst) {
                case some(out): {
                    unsigned char buf[65536];
                    for (;;) {
                        long long n = in.read(&buf[0], 65536UL);
                        if (n < 0LL) { ok = 0; break; }
                        if (n == 0LL) { break; }
                        if (out.write(&buf[0], (unsigned long)n) != n) { ok = 0; break; }
                    }
                    if (out.close() != 0) { ok = 0; }
                }
                case null: { ok = 0; }
            }
            in.close();
            return ok;
        }
        case null: { return 0; }
    }
    return 0;
}

static int dir_cmp_string_(const &String a, const &String b) {
    return a.cmp(b);
}

// Appends 'prefix'-relative paths of the files below 'dir' to 'out'.
static int dir_walk_into_(const &char dir, const &char prefix, &Vec<struct String> out) {
    ?&Dir d = Dir::open(dir);
    int ok = 1;
    match (d) {
        case some(h): {
            struct DirEntry e = DirEntry::new();
            while (h.next(&e)) {
                struct String rel = path_join(prefix, e.name.as_ptr());
                if (e.kind == DIR_DIR) {
                    struct String full = path_join(dir, e.name.as_ptr());
                    if (!dir_walk_into_(full.as_ptr(), rel.as_ptr(), out)) { ok = 0; }
                    full.free();
                    rel.free();
                } else if (e.kind == DIR_LINK) {
                    // A link to a file is listed; a link to a directory is not followed.
                    struct String full = path_join(dir, e.name.as_ptr());
                    ?&Dir probe = Dir::open(full.as_ptr());
                    match (probe) {
                        case some(p): { p.close(); rel.free(); }
                        case null: { out.push(rel); }
                    }
                    full.free();
                } else {
                    out.push(rel);
                }
            }
            e.free();
            h.close();
        }
        case null: { ok = 0; }
    }
    return ok;
}

int dir_walk(const &char root, &Vec<struct String> out) {
    unsigned long first = out.len;
    int ok = dir_walk_into_(root, "", out);
    // Sort just the entries this call added.
    struct Vec<struct String> added = Vec::new();
    while (out.len > first) {
        ?struct String s = out.pop();
        match (s) { case some(v): { added.push(v); } case none: { } }
    }
    added.sort(dir_cmp_string_);
    for (unsigned long i = 0UL; i < added.len; i = i + 1UL) { out.push(*added.at(i)); }
    added.free();
    return ok;
}

int dir_copy_tree(const &char from, const &char to) {
    if (!dir_create_all(to)) { return 0; }
    int ok = 1;
    ?&Dir d = Dir::open(from);
    match (d) {
        case some(h): {
            struct DirEntry e = DirEntry::new();
            while (h.next(&e)) {
                struct String src = path_join(from, e.name.as_ptr());
                struct String dst = path_join(to, e.name.as_ptr());
                int kind = e.kind;
                if (kind == DIR_LINK || kind == DIR_OTHER) {
                    // Copy what a link points at: a directory tree or a file.
                    ?&Dir probe = Dir::open(src.as_ptr());
                    match (probe) {
                        case some(p): { p.close(); kind = DIR_DIR; }
                        case null: { kind = DIR_FILE; }
                    }
                }
                if (kind == DIR_DIR) {
                    if (!dir_copy_tree(src.as_ptr(), dst.as_ptr())) { ok = 0; }
                } else if (!file_copy(src.as_ptr(), dst.as_ptr())) {
                    ok = 0;
                }
                src.free();
                dst.free();
            }
            e.free();
            h.close();
        }
        case null: { ok = 0; }
    }
    return ok;
}

} // namespace std
