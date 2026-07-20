// SafeC Standard Library — XML serialization (built on std/serial/value.h)
//
// Same Value tree as json.h (see its top comment for how to make a struct
// serializable via to_value()); the mapping to XML, since XML elements are
// always named and JSON's array/object shapes aren't:
//   - VAL_OBJECT under tag T   -> <T><key1>...</key1><key2>...</key2></T>
//     (each object key becomes a child element named after the key)
//   - VAL_ARRAY under tag T    -> <T><item>...</item><item>...</item></T>
//     (array elements have no name of their own, so each becomes <item>)
//   - VAL_STRING/INT/FLOAT/BOOL -> <T>text content</T>
//   - VAL_NULL                 -> <T/>
#pragma once
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Appends v's XML representation (as a single element named 'tag') to 'out'.
void xml_write(const &Value v, const char* tag, &String out);

// Convenience wrapper: a fresh String holding just that one element.
struct String value_to_xml(const &Value v, const char* root_tag);

// Parses this module's own output shape back into a Value tree (root tag
// itself is not returned — only its content, mirroring json_parse). Sets
// *ok to 0 (and returns VAL_NULL) on malformed input. This is a parser for
// xml_write's specific grammar, not general XML (no attributes, CDATA, or
// entities beyond the 5 xml_append_escaped_ emits) — see xml.sc's parser
// section for the two structural ambiguities inherent to this grammar
// (empty string vs. empty object; numeric-looking text vs. string).
struct Value xml_parse(const char* text, int* ok);

} // namespace std
