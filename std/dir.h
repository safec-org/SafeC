// SafeC Standard Library — Directories and file system entries (hosted)
//
// Listing, creating, copying and removing directories, and querying what a
// path names. Backends: dir_posix.sc (Linux, macOS, BSD) and dir_win32.sc
// (Windows); the tree operations (create_all, walk, copy, remove_all) are
// shared, in dir.sc. Paths are '/'-separated (Windows accepts '/' too);
// see std/path.h for joining and normalizing them.
//
//   ?&Dir d = Dir::open("docs");
//   match (d) {
//       case some(dir): {
//           struct DirEntry e = DirEntry::new();
//           while (dir.next(&e)) { println(e.name.as_ptr()); }
//           e.free();
//           dir.close();
//       }
//       case null: { eprintln("cannot open docs/"); }
//   }
//
//   struct Vec<struct String> files = Vec::new();
//   dir_walk("docs", &files);            // every file below docs/, sorted
//   dir_copy_tree("public", "dist");     // cp -R public/. dist/
//
// All int-returning operations return 1 on success and 0 on failure.
#pragma once
#include <std/collections/string.h>
#include <std/collections/vec.h>

// What a path or directory entry names.
#define DIR_NONE  0   // nothing there (file_stat failed)
#define DIR_FILE  1   // a regular file
#define DIR_DIR   2   // a directory
#define DIR_LINK  3   // a symbolic link (not followed) / Windows reparse point
#define DIR_OTHER 4   // device, pipe, socket, ...

namespace std {

struct DirEntry {
    struct String name;   // the entry's own name (no directory part)
    int kind;             // DIR_FILE / DIR_DIR / DIR_LINK / DIR_OTHER

    static struct DirEntry new();
    void drop();
    void free() consume;   // = drop()
};

// An open directory listing.
struct Dir {
    int opaque_;   // never constructed by SafeC code; only referenced

    // Opens 'path' for listing; null if it isn't a readable directory.
    static ?&Dir open(const &char path);

    // Reads the next entry into 'out' (its name is replaced). "." and ".."
    // are skipped. Order is whatever the OS returns. 1 if an entry was
    // read, 0 at the end of the listing.
    int  next(&DirEntry out);

    // Closes the listing (the handle is invalid afterwards).
    void drop();
    void close() consume;   // = drop()
};

// What file_stat reports.
struct FileStat {
    int       kind;      // DIR_NONE .. DIR_OTHER (a link itself, not its target)
    long long size;      // bytes (regular files)
    long long mtime_ns;  // last modification, ns since the Unix epoch
                         // (0 where the platform backend can't tell: see dir_posix.sc)
};

// Fills 'out' for 'path' (symbolic links are not followed). 0, with
// out.kind = DIR_NONE, if nothing is there.
int file_stat(const &char path, &FileStat out);

// The kind of 'path' (DIR_NONE when nothing is there).
int path_kind(const &char path);

// 1 if something exists at 'path'.
int path_exists(const &char path);

// Creates one directory (its parent must exist). 1 also when it already
// exists as a directory.
int dir_create(const &char path);

// Creates 'path' and every missing parent (mkdir -p).
int dir_create_all(const &char path);

// Removes an empty directory.
int dir_remove(const &char path);

// Removes 'path' and everything below it (rm -rf); links are removed, not
// followed. 1 also when nothing was there.
int dir_remove_all(const &char path);

// Deletes a file (or a link).
int file_remove(const &char path);

// Copies the file 'from' to 'to' (created or truncated).
int file_copy(const &char from, const &char to);

// Appends the path of every file below 'root', relative to it and
// '/'-separated ("guide/intro.md"), to 'out', sorted. Directories are
// descended into (links to them are not); hidden entries are included.
int dir_walk(const &char root, &Vec<struct String> out);

// Copies everything inside 'from' into 'to' (cp -R from/. to/), creating
// 'to' and subdirectories as needed; existing files are overwritten.
int dir_copy_tree(const &char from, const &char to);

} // namespace std
