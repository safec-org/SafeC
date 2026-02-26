// SafeC Standard Library — HashMap implementation
#include "map.h"
#include "../mem.h"
#include "../str.h"

// ── Hash functions ────────────────────────────────────────────────────────────
unsigned int map_hash_bytes_(const void* data, unsigned long len) {
    unsigned int h = 5381U;
    unsigned long i = 0UL;
    unsafe {
        const char* p = (const char*)data;
        while (i < len) {
            h = ((h << 5) + h) + (unsigned int)(unsigned char)p[i];
            i = i + 1UL;
        }
    }
    return h;
}

unsigned int map_hash_str_(const char* s) {
    unsigned int h = 5381U;
    unsafe {
        while (*s != 0) {
            h = ((h << 5) + h) + (unsigned int)(unsigned char)(*s);
            s = s + 1;
        }
    }
    return h;
}

// ── Internal table allocator ──────────────────────────────────────────────────
struct HashMap map_alloc_table_(unsigned long key_size, unsigned long val_size, unsigned long cap) {
    struct HashMap m;
    m.key_size = key_size; m.val_size = val_size;
    m.len = 0UL; m.cap = cap;
    unsafe {
        m.buckets = (struct MapEntry*)alloc_zeroed(cap * sizeof(struct MapEntry));
    }
    if (m.buckets == (struct MapEntry*)0) { m.cap = 0UL; }
    return m;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct HashMap map_new(unsigned long key_size, unsigned long val_size) {
    return map_alloc_table_(key_size, val_size, 16UL);
}

struct HashMap map_with_cap(unsigned long key_size, unsigned long val_size, unsigned long cap) {
    unsigned long c = 16UL;
    while (c < cap) { c = c * 2UL; }
    return map_alloc_table_(key_size, val_size, c);
}

void HashMap::free_entries_() {
    if (self.buckets == (struct MapEntry*)0) { return; }
    unsigned long i = 0UL;
    while (i < self.cap) {
        if (self.buckets[i].state == 1) {
            unsafe {
                if (self.buckets[i].key != (void*)0) { dealloc(self.buckets[i].key); }
                if (self.buckets[i].val != (void*)0) { dealloc(self.buckets[i].val); }
            }
        }
        i = i + 1UL;
    }
}

void HashMap::free() {
    self.free_entries_();
    unsafe { if (self.buckets != (struct MapEntry*)0) { dealloc((void*)self.buckets); } }
    self.buckets = (struct MapEntry*)0;
    self.cap = 0UL; self.len = 0UL;
}

void HashMap::clear() {
    self.free_entries_();
    unsafe { safe_memset((void*)self.buckets, 0, self.cap * sizeof(struct MapEntry)); }
    self.len = 0UL;
}

unsigned long HashMap::length() const { return self.len; }
int           HashMap::is_empty() const { return self.len == 0UL; }

// ── Resize ────────────────────────────────────────────────────────────────────
int HashMap::resize_(unsigned long new_cap) {
    struct HashMap nm = map_alloc_table_(self.key_size, self.val_size, new_cap);
    if (nm.buckets == (struct MapEntry*)0) { return 0; }
    unsigned long i = 0UL;
    while (i < self.cap) {
        if (self.buckets[i].state == 1) {
            unsigned int h   = self.buckets[i].hash;
            unsigned long idx = (unsigned long)(h & (unsigned int)(new_cap - 1UL));
            unsigned long j   = 0UL;
            while (nm.buckets[idx].state == 1) {
                idx = (idx + 1UL) % new_cap;
                j = j + 1UL;
            }
            nm.buckets[idx] = self.buckets[i];
            nm.len = nm.len + 1UL;
        }
        i = i + 1UL;
    }
    unsafe { dealloc((void*)self.buckets); }
    self.buckets = nm.buckets;
    self.cap = new_cap;
    return 1;
}

// ── Core operations ───────────────────────────────────────────────────────────
int HashMap::insert(const void* key, const void* val) {
    if (self.len * 4UL >= self.cap * 3UL) {
        if (!self.resize_(self.cap * 2UL)) { return 0; }
    }
    unsigned int h = map_hash_bytes_(key, self.key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(self.cap - 1UL));
    unsigned long i = 0UL;
    while (i < self.cap) {
        int s = self.buckets[idx].state;
        if (s == 0 || s == 2) {
            unsafe {
                if (self.buckets[idx].key == (void*)0)
                    self.buckets[idx].key = alloc(self.key_size);
                if (self.buckets[idx].val == (void*)0)
                    self.buckets[idx].val = alloc(self.val_size);
                if (self.buckets[idx].key == (void*)0 || self.buckets[idx].val == (void*)0) { return 0; }
                safe_memcpy(self.buckets[idx].key, key, self.key_size);
                safe_memcpy(self.buckets[idx].val, val, self.val_size);
            }
            self.buckets[idx].hash  = h;
            self.buckets[idx].state = 1;
            self.len = self.len + 1UL;
            return 1;
        }
        if (s == 1 && self.buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)self.buckets[idx].key, key, self.key_size) == 0;
            if (eq) {
                unsafe { safe_memcpy(self.buckets[idx].val, val, self.val_size); }
                return 1;
            }
        }
        idx = (idx + 1UL) % self.cap;
        i = i + 1UL;
    }
    return 0;
}

void* HashMap::get(const void* key) const {
    if (self.len == 0UL) { return (void*)0; }
    unsigned int h = map_hash_bytes_(key, self.key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(self.cap - 1UL));
    unsigned long i = 0UL;
    while (i < self.cap) {
        int s = self.buckets[idx].state;
        if (s == 0) { return (void*)0; }
        if (s == 1 && self.buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)self.buckets[idx].key, key, self.key_size) == 0;
            if (eq) { return self.buckets[idx].val; }
        }
        idx = (idx + 1UL) % self.cap;
        i = i + 1UL;
    }
    return (void*)0;
}

int HashMap::contains(const void* key) const {
    return self.get(key) != (void*)0;
}

int HashMap::remove(const void* key) {
    if (self.len == 0UL) { return 0; }
    unsigned int h = map_hash_bytes_(key, self.key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(self.cap - 1UL));
    unsigned long i = 0UL;
    while (i < self.cap) {
        int s = self.buckets[idx].state;
        if (s == 0) { return 0; }
        if (s == 1 && self.buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)self.buckets[idx].key, key, self.key_size) == 0;
            if (eq) {
                self.buckets[idx].state = 2; // tombstone
                self.len = self.len - 1UL;
                return 1;
            }
        }
        idx = (idx + 1UL) % self.cap;
        i = i + 1UL;
    }
    return 0;
}

void HashMap::foreach(void* fn) {
    unsafe {
        void (*f)(const void*, void*) = (void (*)(const void*, void*))fn;
        unsigned long i = 0UL;
        while (i < self.cap) {
            if (self.buckets[i].state == 1)
                f((const void*)self.buckets[i].key, self.buckets[i].val);
            i = i + 1UL;
        }
    }
}

// ── String-keyed map ──────────────────────────────────────────────────────────
struct HashMap str_map_new(unsigned long val_size) {
    return map_alloc_table_(8UL, val_size, 16UL);
}

int str_map_insert(struct HashMap* m, const char* key, const void* val) {
    if (m->len * 4UL >= m->cap * 3UL) {
        if (!m->resize_(m->cap * 2UL)) { return 0; }
    }
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0 || s == 2) {
            unsafe {
                unsigned long klen = str_len(key) + 1UL;
                char* kcopy = (char*)alloc(klen);
                if (kcopy == (char*)0) { return 0; }
                safe_memcpy((void*)kcopy, (const void*)key, klen);
                if (s == 1 && m->buckets[idx].key != (void*)0)
                    dealloc(m->buckets[idx].key);
                m->buckets[idx].key = (void*)kcopy;
                if (m->buckets[idx].val == (void*)0)
                    m->buckets[idx].val = alloc(m->val_size);
                if (m->buckets[idx].val == (void*)0) { return 0; }
                safe_memcpy(m->buckets[idx].val, val, m->val_size);
            }
            m->buckets[idx].hash  = h;
            m->buckets[idx].state = 1;
            m->len = m->len + 1UL;
            return 1;
        }
        if (s == 1 && m->buckets[idx].hash == h) {
            unsafe {
                if (str_cmp((const char*)m->buckets[idx].key, key) == 0) {
                    safe_memcpy(m->buckets[idx].val, val, m->val_size);
                    return 1;
                }
            }
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return 0;
}

void* str_map_get(struct HashMap* m, const char* key) {
    if (m->len == 0UL) { return (void*)0; }
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) { return (void*)0; }
        if (s == 1 && m->buckets[idx].hash == h) {
            unsafe {
                if (str_cmp((const char*)m->buckets[idx].key, key) == 0)
                    return m->buckets[idx].val;
            }
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return (void*)0;
}

int str_map_contains(struct HashMap* m, const char* key) {
    return str_map_get(m, key) != (void*)0;
}

int str_map_remove(struct HashMap* m, const char* key) {
    if (m->len == 0UL) { return 0; }
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) { return 0; }
        if (s == 1 && m->buckets[idx].hash == h) {
            unsafe {
                if (str_cmp((const char*)m->buckets[idx].key, key) == 0) {
                    dealloc(m->buckets[idx].key);
                    m->buckets[idx].key   = (void*)0;
                    m->buckets[idx].state = 2;
                    m->len = m->len - 1UL;
                    return 1;
                }
            }
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return 0;
}

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int map_insert_t(&stack HashMap m, const void* key, T val) {
    return m.insert(key, (const void*)&val);
}

generic<T>
T* map_get_t(&stack HashMap m, const void* key) {
    return (T*)m.get(key);
}
