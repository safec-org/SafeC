// SafeC Standard Library — HTML serialization (built on std/serial/value.h)
//
// Same Value tree as json.h/xml.h. Unlike JSON/XML, HTML isn't itself a
// data-interchange format, so 'html_write' renders a Value tree as a
// human-readable *document fragment* rather than something meant to be
// parsed back:
//   - VAL_OBJECT  -> <dl><dt>key</dt><dd>...</dd>...</dl>  (definition list)
//   - VAL_ARRAY   -> <ul><li>...</li>...</ul>
//   - VAL_STRING/INT/FLOAT/BOOL -> escaped text
//   - VAL_NULL    -> <em>null</em>
// This is deliberately the same shape as json_write/xml_write (a Value
// tree in, escaped markup text out) for consistency across the three
// format backends, even though "parse HTML back into a Value" isn't a
// meaningful operation the way JSON/XML decoding is.
#pragma once
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Escapes '&', '<', '>', '"' for safe inclusion in HTML text/attribute
// content — useful on its own, independent of the Value-tree renderer
// below, for anything that needs to drop untrusted text into HTML.
void html_escape(struct String* out, const char* s);

// Appends v's rendered HTML fragment to 'out'.
void html_write(const struct Value* v, struct String* out);

// Convenience wrapper: a fresh String holding just that fragment.
struct String value_to_html(const struct Value* v);

// Parses this module's own <dl>/<ul>/<em>null</em>/text fragment shape
// back into a Value tree. Sets *ok to 0 (and returns VAL_NULL) on
// malformed input. Not a general HTML parser — see html.sc's parser
// section for the two structural ambiguities inherent to this grammar
// (same as xml_parse's: empty string vs. empty container; numeric-looking
// text vs. string).
struct Value html_parse(const char* text, int* ok);

} // namespace std
