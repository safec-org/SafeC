// SafeC Standard Library — Protocol Buffers wire format implementation
// (see protobuf.h)
#pragma once
#include <std/serial/protobuf.h>
#include <std/collections/string.h>
#include <std/mem.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/str.h>

// Float <-> bit-pattern copies go through memcpy, not a pointer cast:
// strict aliasing (see safec-docs reference/safety) forbids reading a
// double through an integer lvalue.
#include <std/stddef.h>
extern void* memcpy(void* dst, const void* src, size_t n);   // size_t: 32-bit on wasm32 / 32-bit targets

namespace std {

// ── Writer ──────────────────────────────────────────────────────────────────

void pb_write_varint(&String out, unsigned long long v) {
    while (v >= 0x80ULL) {
        unsigned char b = (unsigned char)((v & 0x7FULL) | 0x80ULL);
        unsafe { out.push_char((char)b); }
        v = v >> 7;
    }
    unsafe { out.push_char((char)(unsigned char)v); }
}

void pb_write_tag(&String out, int field_num, int wire_type) {
    unsigned long long key = ((unsigned long long)(unsigned int)field_num << 3) |
                              (unsigned long long)(unsigned int)wire_type;
    pb_write_varint(out, key);
}

void pb_write_int64_field(&String out, int field_num, long long v) {
    pb_write_tag(out, field_num, PB_WIRE_VARINT);
    pb_write_varint(out, (unsigned long long)v); // sign-extends, matching protoc
}

void pb_write_uint64_field(&String out, int field_num, unsigned long long v) {
    pb_write_tag(out, field_num, PB_WIRE_VARINT);
    pb_write_varint(out, v);
}

void pb_write_sint64_field(&String out, int field_num, long long v) {
    pb_write_tag(out, field_num, PB_WIRE_VARINT);
    unsigned long long zigzag = ((unsigned long long)(v << 1)) ^ (unsigned long long)(v >> 63);
    pb_write_varint(out, zigzag);
}

void pb_write_bool_field(&String out, int field_num, int v) {
    pb_write_tag(out, field_num, PB_WIRE_VARINT);
    pb_write_varint(out, v != 0 ? 1ULL : 0ULL);
}

void pb_write_double_field(&String out, int field_num, double v) {
    pb_write_tag(out, field_num, PB_WIRE_FIXED64);
    unsigned long long bits = 0ULL;
    unsafe { memcpy((void*)&bits, (const void*)&v, (size_t)8UL); }
    int i = 0;
    while (i < 8) {
        unsigned char b = (unsigned char)((bits >> (i * 8)) & 0xFFULL);
        unsafe { out.push_char((char)b); }
        i = i + 1;
    }
}

void pb_write_float_field(&String out, int field_num, float v) {
    pb_write_tag(out, field_num, PB_WIRE_FIXED32);
    unsigned int bits = 0U;
    unsafe { memcpy((void*)&bits, (const void*)&v, (size_t)4UL); }
    int i = 0;
    while (i < 4) {
        unsigned char b = (unsigned char)((bits >> (i * 8)) & 0xFFU);
        unsafe { out.push_char((char)b); }
        i = i + 1;
    }
}

void pb_write_string_field(&String out, int field_num, const &char s) {
    unsigned long len = str_len(s);
    pb_write_tag(out, field_num, PB_WIRE_LEN);
    pb_write_varint(out, (unsigned long long)len);
    unsafe { out.push_n(s, len); }
}

void pb_write_bytes_field(&String out, int field_num, []const unsigned char data) {
    pb_write_tag(out, field_num, PB_WIRE_LEN);
    pb_write_varint(out, (unsigned long long)data.len);
    out.push_bytes(data);
}

// ── Reader ────────────────────────────────────────────────────────────────

struct PbReader PbReader::new([]const unsigned char data) {
    struct PbReader r;
    r.data = data;
    r.pos  = 0UL;
    return r;
}

int PbReader::has_more() const {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = self.pos; len = self.data.len; }
    return pos < len;
}

int PbReader::read_varint(&unsigned long long out) {
    unsigned long long result = 0ULL;
    int shift = 0;
    while (1) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = self.pos; len = self.data.len; }
        if (pos >= len) { return 0; } // truncated
        if (shift >= 64) { return 0; } // malformed: too many continuation bytes
        unsigned char b;
        unsafe { b = self.data[pos]; }
        unsafe { self.pos = pos + 1UL; }
        result = result | (((unsigned long long)(b & 0x7FU)) << shift);
        if (((unsigned int)b & 0x80U) == 0U) { break; }
        shift = shift + 7;
    }
    unsafe { *out = result; }
    return 1;
}

int PbReader::read_tag(&int field_num, &int wire_type) {
    unsigned long long key;
    if (!self.read_varint(&key)) { return 0; }
    unsafe {
        *wire_type = (int)(key & 0x7ULL);
        *field_num = (int)(key >> 3);
    }
    return 1;
}

int PbReader::read_int64(&long long out) {
    unsigned long long v;
    if (!self.read_varint(&v)) { return 0; }
    unsafe { *out = (long long)v; }
    return 1;
}

int PbReader::read_uint64(&unsigned long long out) {
    return self.read_varint(out);
}

int PbReader::read_sint64(&long long out) {
    unsigned long long v;
    if (!self.read_varint(&v)) { return 0; }
    long long decoded = (long long)((v >> 1) ^ (0ULL - (v & 1ULL)));
    unsafe { *out = decoded; }
    return 1;
}

int PbReader::read_bool(&int out) {
    unsigned long long v;
    if (!self.read_varint(&v)) { return 0; }
    unsafe { *out = (v != 0ULL) ? 1 : 0; }
    return 1;
}

int PbReader::read_double(&double out) {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = self.pos; len = self.data.len; }
    if (pos + 8UL > len) { return 0; }
    unsigned long long bits = 0ULL;
    int i = 0;
    while (i < 8) {
        unsigned char b;
        unsafe { b = self.data[pos + (unsigned long)i]; }
        bits = bits | (((unsigned long long)b) << (i * 8));
        i = i + 1;
    }
    unsafe { self.pos = pos + 8UL; }
    double d = 0.0;
    unsafe { memcpy((void*)&d, (const void*)&bits, (size_t)8UL); }
    unsafe { *out = d; }
    return 1;
}

int PbReader::read_float(&float out) {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = self.pos; len = self.data.len; }
    if (pos + 4UL > len) { return 0; }
    unsigned int bits = 0U;
    int i = 0;
    while (i < 4) {
        unsigned char b;
        unsafe { b = self.data[pos + (unsigned long)i]; }
        bits = bits | (((unsigned int)b) << (i * 8));
        i = i + 1;
    }
    unsafe { self.pos = pos + 4UL; }
    float f = (float)0.0;
    unsafe { memcpy((void*)&f, (const void*)&bits, (size_t)4UL); }
    unsafe { *out = f; }
    return 1;
}

?[]const unsigned char PbReader::read_length_delimited() {
    unsigned long long len64;
    if (!self.read_varint(&len64)) { return null; }
    unsigned long fieldLen = (unsigned long)len64;
    unsigned long pos = self.pos;
    if (fieldLen > self.data.len - pos) { return null; } // declared length overruns the buffer
    self.pos = pos + fieldLen;
    return self.data[pos..pos + fieldLen];
}

int PbReader::skip_field(int wire_type) {
    if (wire_type == PB_WIRE_VARINT) {
        unsigned long long v;
        return self.read_varint(&v);
    }
    if (wire_type == PB_WIRE_FIXED64) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = self.pos; len = self.data.len; }
        if (pos + 8UL > len) { return 0; }
        unsafe { self.pos = pos + 8UL; }
        return 1;
    }
    if (wire_type == PB_WIRE_FIXED32) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = self.pos; len = self.data.len; }
        if (pos + 4UL > len) { return 0; }
        unsafe { self.pos = pos + 4UL; }
        return 1;
    }
    if (wire_type == PB_WIRE_LEN) {
        return self.read_length_delimited().is_none() ? 0 : 1;
    }
    return 0; // wire type 3/4 (deprecated groups) — not supported
}

} // namespace std
