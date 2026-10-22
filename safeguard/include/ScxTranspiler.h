#pragma once
#include <string>
#include <vector>

namespace safeguard {

// Transpiles a .scx file (ordinary SafeC source, plus HTML-like markup in
// three statement positions) into plain SafeC source that 'safec' can
// compile unmodified. Markup may be:
//
//   return <markup>;                 // the function's String result
//   struct String s = <markup>;      // a declaration's initializer
//   lhs = <markup>;                  // an assignment (any lvalue)
//
// where <markup> is one element or a fragment — a sequence of elements,
// '<!doctype ...>' declarations and '<!-- comments -->' (kept verbatim):
//
//   <tag attr="literal" attr2={expr}>
//     text
//     {expr}      // HTML-escaped and appended
//     {!expr}     // appended raw/unescaped (e.g. pre-rendered child HTML)
//     <nested>...</nested>
//   </tag>
//
// 'tag' names identifiers ([a-zA-Z][a-zA-Z0-9-]*); a tag with no children
// may self-close as <tag ... />. Output follows HTML rules: an empty void
// element (br, img, meta, link, ...) is written as a start tag only, any
// other empty element gets its close tag ('<div></div>' — '<div/>' would
// parse as an open tag), and only inside <svg>/<math> are empty elements
// written self-closed. <script> and <style> hold raw text: everything up
// to the close tag is copied verbatim, braces included, with no
// interpolation. {expr}/{!expr} expressions must evaluate
// to 'const char*' — scx does no type-checking of its own (it runs before
// safec's Sema ever sees the file), so a type mismatch surfaces as an
// ordinary safec compile error on the generated code, pointing at the
// synthesized .push()/scx_append_esc() call site.
//
// A quoted '#include "x.h"' is rewritten to the absolute path of x.h next
// to the .scx file (when it exists there), since the generated file is
// compiled from another directory.
//
// Everything outside a 'return <markup>;' statement passes through
// byte-for-byte untouched (still ordinary SafeC — imports, structs, plain
// functions, etc.) — a .scx file's non-markup majority is exactly a .sc
// file. String/char literals and // and /* */ comments are scanned over
// verbatim so 'return' or '<' appearing inside them is never mistaken for
// markup.
//
// Throws std::runtime_error (message prefixed "<file>:<line>: ...") on
// malformed markup: unterminated tag, mismatched open/close tag names, an
// unterminated {expr}, or markup that never resolves to a complete,
// well-formed element followed by ';'.
std::string transpileScx(const std::string& source, const std::string& filename);

// Same transpilation, plus a line map for tools (the LSP) that need to
// translate a position in the *generated* SafeC source back to the
// original .scx buffer — e.g. to point a diagnostic or a hover result at
// the right spot in the file the user is actually editing, since safec
// itself only ever sees the generated text.
//
// 'lineMap' is filled with one entry per generated line, 1-indexed
// (lineMap[0] is unused/zero; lineMap[1] is the original line generated
// line 1 corresponds to, and so on — size() == generated line count + 1).
// Mapping is exact for every line outside a 'return <markup>;' expansion
// (those are copied byte-for-byte, so original and generated lines
// coincide 1:1); every line *inside* an expansion — including the two
// '#include' lines auto-prepended when the file used any markup — maps to
// the original line the expansion started on (its 'return' keyword, or
// line 1 for the prepended includes), since those lines have no true 1:1
// original counterpart.
std::string transpileScx(const std::string& source, const std::string& filename,
                          std::vector<int>* lineMap);

} // namespace safeguard
