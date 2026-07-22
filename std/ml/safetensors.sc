// SafeC Standard Library — safetensors weight file reader implementation
// (see safetensors.h).
#pragma once
#include <std/ml/safetensors.h>
#include <std/ml/float16.h>
#include <std/serial/json.h>
#include <std/io_file.h>
#include <std/mem.h>
#include <std/str.h>

namespace std {

extern int printf(const char* fmt, ...);

static unsigned long long __st_read_u64_le(void* f) {
    unsigned char buf[8];
    unsigned long got;
    unsafe { got = file_read(f, (void*)&buf[0], 8UL); }
    if (got != 8UL) { return 0xFFFFFFFFFFFFFFFFULL; }
    unsigned long long v = 0ULL;
    int i = 7;
    while (i >= 0) {
        unsafe { v = (v << 8) | (unsigned long long)buf[i]; }
        i = i - 1;
    }
    return v;
}

long SafetensorsFile::find(const char* name) const {
    unsigned long i = 0UL;
    while (i < self.entries.length()) {
        unsafe {
            struct SafetensorsEntry* e = (struct SafetensorsEntry*)self.entries.get_raw(i);
            if (str_eq(e->name.as_ptr(), name)) { return (long)i; }
        }
        i = i + 1UL;
    }
    return -1L;
}

struct Tensor* SafetensorsFile::load(const char* name) {
    long idx = self.find(name);
    if (idx < 0L) {
        printf("safetensors: tensor not found: %s\n", name);
        return (struct Tensor*)0;
    }
    struct SafetensorsEntry* e;
    unsigned long byteLen;
    unsafe {
        e = (struct SafetensorsEntry*)self.entries.get_raw((unsigned long)idx);
        byteLen = e->dataOffsetEnd - e->dataOffsetStart;
        file_seek(self.handle, (long long)(self.dataBaseOffset + e->dataOffsetStart), 0);
    }
    unsigned char* raw = (unsigned char*)malloc(byteLen);
    unsigned long got;
    unsafe { got = file_read(self.handle, (void*)raw, byteLen); }
    if (got != byteLen) {
        printf("safetensors: short read for %s (wanted %lu, got %lu)\n", name, byteLen, got);
        unsafe { free((void*)raw); }
        return (struct Tensor*)0;
    }

    struct Tensor* out;
    int isF32; int isBF16;
    unsafe {
        out = __tensor_alloc((const unsigned long*)&e->shape[0], e->ndim, 0);
        isF32 = str_eq(e->dtype.as_ptr(), "F32");
        isBF16 = str_eq(e->dtype.as_ptr(), "BF16");
    }

    if (isF32) {
        unsafe {
            if (byteLen != out->size * 4UL) {
                printf("safetensors: F32 size mismatch for %s\n", name);
            }
            unsigned long b = 0UL;
            while (b < out->size) {
                ((float*)out->data)[b] = ((float*)raw)[b];
                b = b + 1UL;
            }
        }
    } else if (isBF16) {
        unsafe {
            bf16_to_f32_bulk((const unsigned short*)raw, (float*)out->data, out->size);
        }
    } else {
        unsafe {
            printf("safetensors: unsupported dtype '%s' for %s (only F32/BF16 handled)\n",
                   e->dtype.as_ptr(), name);
            free((void*)raw);
            out->free();
        }
        return (struct Tensor*)0;
    }

    unsafe { free((void*)raw); }
    return out;
}

void SafetensorsFile::close_() {
    if (self.handle != (void*)0) {
        file_close(self.handle);
        self.handle = (void*)0;
    }
    unsigned long i = 0UL;
    while (i < self.entries.length()) {
        unsafe {
            struct SafetensorsEntry* e = (struct SafetensorsEntry*)self.entries.get_raw(i);
            e->name.free();
            e->dtype.free();
        }
        i = i + 1UL;
    }
    self.entries.free();
}

struct SafetensorsFile safetensors_open(const char* path) {
    struct SafetensorsFile st;
    st.handle = (void*)0;
    st.dataBaseOffset = 0UL;
    st.entries = vec_new(sizeof(struct SafetensorsEntry));

    void* f;
    unsafe { f = file_open(path, "rb"); }
    if (f == (void*)0) {
        printf("safetensors: could not open %s\n", path);
        return st;
    }

    unsigned long long headerLen = __st_read_u64_le(f);
    if (headerLen == 0xFFFFFFFFFFFFFFFFULL || headerLen == 0ULL) {
        printf("safetensors: bad header length in %s\n", path);
        file_close(f);
        return st;
    }

    char* headerBuf = (char*)malloc((unsigned long)headerLen + 1UL);
    unsigned long got;
    unsafe { got = file_read(f, (void*)headerBuf, (unsigned long)headerLen); }
    if (got != (unsigned long)headerLen) {
        printf("safetensors: short header read in %s\n", path);
        unsafe { free((void*)headerBuf); }
        file_close(f);
        return st;
    }
    unsafe { headerBuf[headerLen] = (char)0; }

    int ok = 0;
    struct Value header = json_parse(headerBuf, &ok);
    unsafe { free((void*)headerBuf); }
    if (!ok) {
        printf("safetensors: header JSON parse failed in %s\n", path);
        file_close(f);
        return st;
    }

    unsigned long i = 0UL;
    while (i < header.obj_val.length()) {
        unsafe {
            struct ObjectEntry* oe = (struct ObjectEntry*)header.obj_val.get_raw(i);
            const char* key = (const char*)oe->key;
            if (key != (const char*)0 && str_eq(key, "__metadata__") == 0) {
                struct SafetensorsEntry se;
                se.name = string_from(key);

                struct Value* dtypeV = oe->val.object_get("dtype");
                se.dtype = string_from(dtypeV->as_string());

                struct Value* shapeV = oe->val.object_get("shape");
                unsigned long nd = shapeV->array_len();
                if (nd > 8UL) {
                    printf("safetensors: tensor '%s' has rank %lu > 8, truncating\n", key, nd);
                    nd = 8UL;
                }
                se.ndim = nd;
                unsigned long d = 0UL;
                while (d < nd) {
                    &Value elemV = shapeV->array_at(d);
                    se.shape[d] = (unsigned long)elemV.as_int();
                    d = d + 1UL;
                }

                struct Value* offV = oe->val.object_get("data_offsets");
                &Value off0 = offV->array_at(0UL);
                &Value off1 = offV->array_at(1UL);
                se.dataOffsetStart = (unsigned long)off0.as_int();
                se.dataOffsetEnd   = (unsigned long)off1.as_int();

                st.entries.push((const void*)&se);
            }
        }
        i = i + 1UL;
    }
    value_free(&header);

    st.handle = f;
    st.dataBaseOffset = 8UL + (unsigned long)headerLen;
    return st;
}

} // namespace std
