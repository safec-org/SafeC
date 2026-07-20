// SafeC Standard Library — Protocol Buffers wire format implementation
// (see protobuf.h)
#pragma once
#include <std/serial/protobuf.h>
#include <std/collections/string.sc>
#include <std/mem.sc>
#include <std/fmt.sc>
#include <std/convert.sc>

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
    unsigned long long bits;
    unsafe { bits = *(unsigned long long*)&v; }
    int i = 0;
    while (i < 8) {
        unsigned char b = (unsigned char)((bits >> (i * 8)) & 0xFFULL);
        unsafe { out.push_char((char)b); }
        i = i + 1;
    }
}

void pb_write_float_field(&String out, int field_num, float v) {
    pb_write_tag(out, field_num, PB_WIRE_FIXED32);
    unsigned int bits;
    unsafe { bits = *(unsigned int*)&v; }
    int i = 0;
    while (i < 4) {
        unsigned char b = (unsigned char)((bits >> (i * 8)) & 0xFFU);
        unsafe { out.push_char((char)b); }
        i = i + 1;
    }
}

void pb_write_string_field(&String out, int field_num, const char* s) {
    unsigned long len = str_len(s);
    pb_write_tag(out, field_num, PB_WIRE_LEN);
    pb_write_varint(out, (unsigned long long)len);
    unsafe { out.push_n(s, len); }
}

void pb_write_bytes_field(&String out, int field_num,
                           const unsigned char* data, unsigned long len) {
    pb_write_tag(out, field_num, PB_WIRE_LEN);
    pb_write_varint(out, (unsigned long long)len);
    unsafe { out.push_n((const char*)data, len); }
}

// ── Reader ────────────────────────────────────────────────────────────────

struct PbReader pb_reader_new(const unsigned char* data, unsigned long len) {
    struct PbReader r;
    r.data = data;
    r.len  = len;
    r.pos  = 0UL;
    return r;
}

int pb_reader_has_more(const &PbReader r) {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = r.pos; len = r.len; }
    return pos < len;
}

int pb_read_varint(&PbReader r, unsigned long long* out) {
    unsigned long long result = 0ULL;
    int shift = 0;
    while (1) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = r.pos; len = r.len; }
        if (pos >= len) { return 0; } // truncated
        if (shift >= 64) { return 0; } // malformed: too many continuation bytes
        unsigned char b;
        unsafe { b = r.data[pos]; }
        unsafe { r.pos = pos + 1UL; }
        result = result | (((unsigned long long)(b & 0x7FU)) << shift);
        if (((unsigned int)b & 0x80U) == 0U) { break; }
        shift = shift + 7;
    }
    unsafe { *out = result; }
    return 1;
}

int pb_read_tag(&PbReader r, int* field_num, int* wire_type) {
    unsigned long long key;
    if (!pb_read_varint(r, &key)) { return 0; }
    unsafe {
        *wire_type = (int)(key & 0x7ULL);
        *field_num = (int)(key >> 3);
    }
    return 1;
}

int pb_read_int64(&PbReader r, long long* out) {
    unsigned long long v;
    if (!pb_read_varint(r, &v)) { return 0; }
    unsafe { *out = (long long)v; }
    return 1;
}

int pb_read_uint64(&PbReader r, unsigned long long* out) {
    return pb_read_varint(r, out);
}

int pb_read_sint64(&PbReader r, long long* out) {
    unsigned long long v;
    if (!pb_read_varint(r, &v)) { return 0; }
    long long decoded = (long long)((v >> 1) ^ (0ULL - (v & 1ULL)));
    unsafe { *out = decoded; }
    return 1;
}

int pb_read_bool(&PbReader r, int* out) {
    unsigned long long v;
    if (!pb_read_varint(r, &v)) { return 0; }
    unsafe { *out = (v != 0ULL) ? 1 : 0; }
    return 1;
}

int pb_read_double(&PbReader r, double* out) {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = r.pos; len = r.len; }
    if (pos + 8UL > len) { return 0; }
    unsigned long long bits = 0ULL;
    int i = 0;
    while (i < 8) {
        unsigned char b;
        unsafe { b = r.data[pos + (unsigned long)i]; }
        bits = bits | (((unsigned long long)b) << (i * 8));
        i = i + 1;
    }
    unsafe { r.pos = pos + 8UL; }
    double d;
    unsafe { d = *(double*)&bits; }
    unsafe { *out = d; }
    return 1;
}

int pb_read_float(&PbReader r, float* out) {
    unsigned long pos;
    unsigned long len;
    unsafe { pos = r.pos; len = r.len; }
    if (pos + 4UL > len) { return 0; }
    unsigned int bits = 0U;
    int i = 0;
    while (i < 4) {
        unsigned char b;
        unsafe { b = r.data[pos + (unsigned long)i]; }
        bits = bits | (((unsigned int)b) << (i * 8));
        i = i + 1;
    }
    unsafe { r.pos = pos + 4UL; }
    float f;
    unsafe { f = *(float*)&bits; }
    unsafe { *out = f; }
    return 1;
}

int pb_read_length_delimited(&PbReader r, const unsigned char** dataOut,
                              unsigned long* lenOut) {
    unsigned long long len64;
    if (!pb_read_varint(r, &len64)) { return 0; }
    unsigned long fieldLen = (unsigned long)len64;
    unsigned long pos;
    unsigned long bufLen;
    unsafe { pos = r.pos; bufLen = r.len; }
    if (pos + fieldLen > bufLen) { return 0; } // declared length overruns the buffer
    unsafe {
        *dataOut = r.data + pos;
        *lenOut  = fieldLen;
        r.pos   = pos + fieldLen;
    }
    return 1;
}

int pb_skip_field(&PbReader r, int wire_type) {
    if (wire_type == PB_WIRE_VARINT) {
        unsigned long long v;
        return pb_read_varint(r, &v);
    }
    if (wire_type == PB_WIRE_FIXED64) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = r.pos; len = r.len; }
        if (pos + 8UL > len) { return 0; }
        unsafe { r.pos = pos + 8UL; }
        return 1;
    }
    if (wire_type == PB_WIRE_FIXED32) {
        unsigned long pos;
        unsigned long len;
        unsafe { pos = r.pos; len = r.len; }
        if (pos + 4UL > len) { return 0; }
        unsafe { r.pos = pos + 4UL; }
        return 1;
    }
    if (wire_type == PB_WIRE_LEN) {
        const unsigned char* d;
        unsigned long n;
        return pb_read_length_delimited(r, &d, &n);
    }
    return 0; // wire type 3/4 (deprecated groups) — not supported
}

} // namespace std
