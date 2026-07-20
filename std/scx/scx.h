#pragma once
// SafeC Standard Library — scx runtime support
//
// scx (.scx files) is safeguard's JSX/TSX/RSX-style HTML-string templating
// language: a .scx file is ordinary SafeC source, plus one extra piece of
// syntax — 'return <tag attr="...">...</tag>;' — which safeguard's
// scx transpiler (ScxTranspiler.cpp) rewrites into plain SafeC calls
// against 'struct String' *before* the file ever reaches safec. This
// header is the small runtime the generated code calls into; it is never
// included by hand-written .scx source (the transpiler injects the
// #include itself whenever a file actually uses markup).
#include <std/collections/string.h>

namespace std {

// Appends 's' to 'buf', escaping '&', '<', '>', '"', '\'' to their HTML
// entity forms. Used for every '{expr}' interpolation site in scx-generated
// code (the default, safe form — mirrors JSX's automatic escaping of
// embedded expressions). Raw/unescaped interpolation uses '{!expr}' in
// scx source, which compiles directly to buf.push(expr) instead.
void scx_append_esc(&String buf, const char* s);

} // namespace std
