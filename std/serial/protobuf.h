// SafeC Standard Library — Protocol Buffers wire format (proto3 binary
// encoding, https://protobuf.dev/programming-guides/encoding/).
//
// This is a wire-format *primitive* library, not a `.proto`-file IDL
// compiler (SafeC has no field-name/field-number reflection to drive code
// generation from a schema — see std/serial/value.h's top comment on the
// same limitation for JSON). Making a struct protobuf-serializable means
// writing 'to_proto()'/'from_proto()' methods by hand that call these
// primitives with the field numbers your `.proto` schema assigns:
//
//   // .proto: message Point { int32 x = 1; int32 y = 2; }
//   struct Point { int x; int y; };
//   struct String Point::to_proto() const {
//       struct String out = string_new();
//       pb_write_int64_field(&out, 1, (long long)self.x);
//       pb_write_int64_field(&out, 2, (long long)self.y);
//       return out;
//   }
//   struct Point Point::from_proto(const unsigned char* data, unsigned long len, int* ok) {
//       struct Point p;
//       p.x = 0; p.y = 0;
//       struct PbReader r = pb_reader_new(data, len);
//       int good = 1;
//       while (pb_reader_has_more(&r)) {
//           int field; int wireType;
//           if (!pb_read_tag(&r, &field, &wireType)) { good = 0; break; }
//           if (field == 1 && wireType == PB_WIRE_VARINT) {
//               long long v; pb_read_int64(&r, &v); p.x = (int)v;
//           } else if (field == 2 && wireType == PB_WIRE_VARINT) {
//               long long v; pb_read_int64(&r, &v); p.y = (int)v;
//           } else if (!pb_skip_field(&r, wireType)) { good = 0; break; }
//       }
//       if (ok != (int*)0) { *ok = good; }
//       return p;
//   }
//
// Interoperates with real protobuf implementations in other languages —
// the wire format itself (varint, tags, zigzag, length-delimited,
// fixed32/fixed64) is exactly what protoc-generated code produces, field
// numbers and wire types are the only thing that has to match by hand
// since there's no shared `.proto` schema driving both sides here.
#pragma once
#include <std/collections/string.h>

namespace std {

// ── Wire types (proto3 encoding spec) ─────────────────────────────────────────
#define PB_WIRE_VARINT  0   // int32/int64/uint32/uint64/sint32/sint64/bool/enum
#define PB_WIRE_FIXED64 1   // fixed64/sfixed64/double
#define PB_WIRE_LEN     2   // string/bytes/embedded messages/packed repeated
#define PB_WIRE_FIXED32 5   // fixed32/sfixed32/float
// Wire types 3 and 4 (deprecated 'group' start/end) are not supported —
// no proto3 schema emits them.

// ── Writer ──────────────────────────────────────────────────────────────────
// Every pb_write_*_field appends its own tag (field_num, wire_type) before
// the value, so a message is built by calling one of these per field, in
// any order, onto the same 'out' buffer.

void pb_write_varint(&String out, unsigned long long v);
void pb_write_tag(&String out, int field_num, int wire_type);

// int32/int64: plain varint of the value's bit pattern (protobuf's own
// quirk: a negative int32 is sign-extended to 64 bits *before* varint
// encoding, so it always takes the full 10 bytes on the wire — this
// matches protoc's own behavior, not a bug).
void pb_write_int64_field(&String out, int field_num, long long v);
void pb_write_uint64_field(&String out, int field_num, unsigned long long v);
// sint32/sint64: zigzag-encoded varint (efficient for small negative
// values, unlike plain int64 above).
void pb_write_sint64_field(&String out, int field_num, long long v);
void pb_write_bool_field(&String out, int field_num, int v);
void pb_write_double_field(&String out, int field_num, double v);
void pb_write_float_field(&String out, int field_num, float v);
// Length-delimited: string, bytes, or an embedded/nested message's own
// already-encoded bytes (encode the submessage with its own to_proto()
// first, then pass the result here).
void pb_write_string_field(&String out, int field_num, const char* s);
void pb_write_bytes_field(&String out, int field_num,
                           const unsigned char* data, unsigned long len);

// ── Reader ────────────────────────────────────────────────────────────────
// A zero-copy cursor over an existing byte buffer — pb_read_length_delimited
// returns a pointer *into* that same buffer (valid only as long as it is).

struct PbReader {
    const unsigned char* data;
    unsigned long        len;
    unsigned long        pos;
};

struct PbReader pb_reader_new(const unsigned char* data, unsigned long len);
int pb_reader_has_more(const &PbReader r);

// Each returns 1 on success, 0 on a truncated/malformed encoding (ran out
// of bytes mid-varint, length-delimited field's declared length exceeds
// the remaining buffer, etc.).
int pb_read_varint(&PbReader r, unsigned long long* out);
int pb_read_tag(&PbReader r, int* field_num, int* wire_type);
int pb_read_int64(&PbReader r, long long* out);
int pb_read_uint64(&PbReader r, unsigned long long* out);
int pb_read_sint64(&PbReader r, long long* out);
int pb_read_bool(&PbReader r, int* out);
int pb_read_double(&PbReader r, double* out);
int pb_read_float(&PbReader r, float* out);
// '*dataOut' aliases into the reader's own buffer (see the zero-copy note
// above) — copy it out (e.g. via string_from_n) if it needs to outlive
// that buffer.
int pb_read_length_delimited(&PbReader r, const unsigned char** dataOut,
                              unsigned long* lenOut);

// Advances past one field's value without decoding it, given its wire
// type from a just-read tag — the standard "ignore fields I don't
// recognize" behavior every protobuf reader needs for forward
// compatibility with schemas that gained fields since this code was
// written. Returns 0 for wire type 3/4 (unsupported group markers) or a
// truncated encoding.
int pb_skip_field(&PbReader r, int wire_type);

} // namespace std
