// SafeC Standard Library — YAML serialization (block style subset), built
// on std/serial/value.h the same way json.h/csv.h/xml.h/html.h are.
//
// Supported: block scalars (plain, single-quoted '...' with '' escaping,
// double-quoted "..." with \" \\ \n \t \r \t escapes), block sequences
// ('- item' per line, including the compact '- key: value' list-of-maps
// shorthand), block mappings ('key: value' per line, indentation-nested),
// '#' comments (outside quotes), and the null/bool/int/float scalar forms
// YAML 1.1 core schema recognizes (~/null, true/false, plain integers and
// floats).
//
// NOT supported (out of scope for this module — a full YAML 1.2 processor
// is a much larger undertaking): flow style ('[a, b]' / '{k: v}'), anchors
// & aliases ('&name' / '*name'), tags ('!!str'), multi-document streams
// ('---' separators), block scalar literals ('|' / '>'), and merge keys
// ('<<:'). A document using any of these either parses as plain scalar
// text (least-surprise fallback) or fails — see yaml_parse's 'ok' contract.
#pragma once
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Appends v's YAML text representation to 'out' (does not clear 'out'
// first). Only VAL_NULL/VAL_BOOL/VAL_INT/VAL_FLOAT/VAL_STRING/VAL_ARRAY/
// VAL_OBJECT are meaningful — same value tree json_write accepts.
void yaml_write(const struct Value* v, struct String* out);

// Convenience wrapper: a fresh String holding just v's YAML text.
struct String value_to_yaml(const struct Value* v);

// Parses a block-style YAML document (see the file-level comment for the
// supported subset). On success, '*ok' (if non-NULL) is set to 1; on a
// structural error (inconsistent indentation, unterminated quoted scalar,
// a line that's neither a valid sequence/mapping entry nor a bare scalar
// where one is expected), '*ok' is set to 0 and the return value is
// whatever was parsed before the error.
struct Value yaml_parse(const char* text, int* ok);

} // namespace std
