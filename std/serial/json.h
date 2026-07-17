// SafeC Standard Library — JSON serialization (built on std/serial/value.h)
//
// Making a struct JSON-serializable: SafeC has no field-name reflection
// (see value.h's top comment), so write a 'to_value()' method by hand and
// let this module handle the rest:
//
//   struct Point { double x; double y; };
//   struct Value Point::to_value() const {
//       struct Value v = value_object();
//       value_object_set(&v, "x", value_float(self.x));
//       value_object_set(&v, "y", value_float(self.y));
//       return v;
//   }
//   ...
//   struct String json = value_to_json(&p.to_value());
//
// (a from_value()/from_json() the same way covers the decode direction)
#pragma once
#include <std/serial/value.h>
#include <std/collections/string.h>

namespace std {

// Appends v's JSON text representation to 'out' (does not clear 'out'
// first — lets you build up a larger document across multiple calls).
void json_write(const struct Value* v, struct String* out);

// Convenience wrapper: a fresh String holding just v's JSON text.
struct String value_to_json(const struct Value* v);

// Parses a JSON document. On success, '*ok' (if non-NULL) is set to 1 and
// the parsed tree is returned; on failure, '*ok' is set to 0 and the
// return value is value_null(). Ignores trailing content after the first
// complete value (matches most JSON libraries' default leniency).
struct Value json_parse(const char* text, int* ok);

} // namespace std
