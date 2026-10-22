// SafeC Standard Library — Path strings
//
// Pure string manipulation of file system paths: nothing here touches the
// file system (see std/dir.h for that). Every function returns a fresh
// String (free it), and accepts '/' as the separator everywhere; on
// Windows targets '\' is a separator too. Results always use '/'
// (Windows APIs accept it).
//
//   struct String p = path_join("site", "../dist/index.html"); // "site/../dist/index.html"
//   struct String n = path_normalize(p.as_ptr());              // "dist/index.html"
//   struct String e = path_ext("a/b.tar.gz");                  // "gz"
#pragma once
#include <std/collections/string.h>

namespace std {

// 1 if 'c' separates path components on this target ('/', and '\' on Windows).
int path_is_sep(char c);

// 1 for "/x", and on Windows also "\x", "C:\x" and "C:/x".
int path_is_absolute(const &char p);

// 'a' + '/' + 'b' with exactly one separator between them; when 'b' is
// absolute the result is just 'b' (like Python's os.path.join). An empty
// side yields the other.
struct String path_join(const &char a, const &char b);

// Everything before the last component: "a/b/c" -> "a/b", "c" -> "",
// "/c" -> "/", "a/b/" -> "a". Trailing separators are ignored.
struct String path_dirname(const &char p);

// The last component: "a/b.c" -> "b.c", "a/b/" -> "b", "/" -> "".
struct String path_basename(const &char p);

// The last component's extension, without the dot: "a/b.tar.gz" -> "gz",
// "b" -> "", ".bashrc" -> "" (a leading dot names a file, not an extension).
struct String path_ext(const &char p);

// The last component without its extension: "a/b.tar.gz" -> "b.tar",
// ".bashrc" -> ".bashrc".
struct String path_stem(const &char p);

// Lexically normalized: repeated separators collapsed, "." components
// dropped, "x/.." pairs removed (leading ".." kept on a relative path,
// dropped at the root of an absolute one), no trailing separator.
// "" and "." normalize to ".".
struct String path_normalize(const &char p);

// The path that leads from directory 'from' to 'to', both normalized
// first: path_relative("a/b", "a/c/d") -> "../c/d"; the same path -> ".".
// Both must be relative, or both absolute (otherwise 'to' is returned,
// normalized).
struct String path_relative(const &char from, const &char to);

} // namespace std
