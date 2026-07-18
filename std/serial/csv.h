// SafeC Standard Library — CSV serialization (RFC 4180), built on
// std/serial/value.h the same way json.h/xml.h/html.h are.
//
// A CSV document is represented as a VAL_ARRAY of rows, each row itself a
// VAL_ARRAY of VAL_STRING fields — CSV has no native typing (every field is
// text), so unlike json_parse there's no int/float/bool inference; convert
// individual fields yourself (String::parse_int()/parse_float(), see
// std/collections/string.h) if a column is known to hold numbers.
//
//   struct Value rows = csv_parse("name,age\nAda,36\n", &ok);
//   struct Value* row0 = rows.array_at(0);
//   struct Value* name = row0->array_at(0);   // VAL_STRING "name"
#pragma once
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Appends v's CSV text representation to 'out' (does not clear 'out' first).
// 'v' must be VAL_ARRAY of VAL_ARRAY of (VAL_STRING/VAL_INT/VAL_FLOAT/
// VAL_BOOL/VAL_NULL) — non-string fields are stringified the same way
// json_write formats them, VAL_NULL becomes an empty field. Fields
// containing a comma, double quote, or newline are quoted per RFC 4180,
// with embedded double quotes doubled ("" ). Rows are separated by "\n".
void csv_write(const struct Value* v, struct String* out);

// Convenience wrapper: a fresh String holding just v's CSV text.
struct String value_to_csv(const struct Value* v);

// Parses a CSV document into a VAL_ARRAY of VAL_ARRAY of VAL_STRING (every
// field, quoted or not, becomes a string — see the file-level comment).
// Accepts both "\n" and "\r\n" line endings; a trailing newline is
// optional. On success, '*ok' (if non-NULL) is set to 1; on a malformed
// quoted field (an opening '"' with no matching close before end of text),
// '*ok' is set to 0 and the return value is whatever was parsed so far.
struct Value csv_parse(const char* text, int* ok);

} // namespace std
