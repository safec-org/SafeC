#pragma once
#include <string>

namespace safeguard {

// Transpiles a .scx file (ordinary SafeC source, plus one extra piece of
// syntax: HTML-like markup as the operand of a 'return' statement) into
// plain SafeC source that 'safec' can compile unmodified.
//
// Supported form (the only place markup may appear — see scx.md in
// safec-docs for the full writeup):
//
//   return <tag attr="literal" attr2={expr}>
//     text
//     {expr}      // HTML-escaped and appended
//     {!expr}     // appended raw/unescaped (e.g. pre-rendered child HTML)
//     <nested>...</nested>
//   </tag>;
//
// 'tag' names identifiers ([a-zA-Z][a-zA-Z0-9-]*); a tag with no children
// may self-close as <tag ... />. {expr}/{!expr} expressions must evaluate
// to 'const char*' — scx does no type-checking of its own (it runs before
// safec's Sema ever sees the file), so a type mismatch surfaces as an
// ordinary safec compile error on the generated code, pointing at the
// synthesized .push()/scx_append_esc() call site.
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
// unterminated {expr}, or a 'return <ident...' that never resolves to a
// complete, well-formed element followed by ';'.
std::string transpileScx(const std::string& source, const std::string& filename);

} // namespace safeguard
