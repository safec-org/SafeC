// SafeC Standard Library — HashMap implementation
#include "map.h"
#include "../mem.h"
#include "../str.h"

// ── Hash function (djb2 byte hash) ────────────────────────────────────────────
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

// ── Internal helpers ──────────────────────────────────────────────────────────
void map_entry_init_(struct MapEntry* e) {
    e->key = (void*)0; e->val = (void*)0; e->hash = 0U; e->state = 0;
}

struct HashMap map_alloc_table_(unsigned long key_size, unsigned long val_size, unsigned long cap) {
    struct HashMap m;
    m.key_size = key_size; m.val_size = val_size;
    m.len = 0UL; m.cap = cap;
    unsafe {
        m.buckets = (struct MapEntry*)alloc_zeroed(cap * sizeof(struct MapEntry));
    }
    if (m.buckets == (struct MapEntry*)0) m.cap = 0UL;
    return m;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct HashMap map_new(unsigned long key_size, unsigned long val_size) {
    return map_alloc_table_(key_size, val_size, 16UL);
}

struct HashMap map_with_cap(unsigned long key_size, unsigned long val_size, unsigned long cap) {
    // Round up to next power of 2
    unsigned long c = 16UL;
    while (c < cap) c = c * 2UL;
    return map_alloc_table_(key_size, val_size, c);
}

void map_free_entries_(struct HashMap* m) {
    if (m->buckets == (struct MapEntry*)0) return;
    unsigned long i = 0UL;
    while (i < m->cap) {
        if (m->buckets[i].state == 1) {
            unsafe {
                if (m->buckets[i].key != (void*)0) dealloc(m->buckets[i].key);
                if (m->buckets[i].val != (void*)0) dealloc(m->buckets[i].val);
            }
        }
        i = i + 1UL;
    }
}

void map_free(struct HashMap* m) {
    map_free_entries_(m);
    unsafe { if (m->buckets != (struct MapEntry*)0) dealloc((void*)m->buckets); }
    m->buckets = (struct MapEntry*)0;
    m->cap = 0UL; m->len = 0UL;
}

void map_clear(struct HashMap* m) {
    map_free_entries_(m);
    unsafe { safe_memset((void*)m->buckets, 0, m->cap * sizeof(struct MapEntry)); }
    m->len = 0UL;
}

unsigned long map_len(struct HashMap* m)    { return m->len; }
int           map_is_empty(struct HashMap* m) { return m->len == 0UL; }

// ── Probe for a slot ──────────────────────────────────────────────────────────
unsigned long map_probe_(struct HashMap* m, unsigned int h, const void* key) {
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) return idx; // empty
        if (s == 1 && m->buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)m->buckets[idx].key, key, m->key_size) == 0;
            if (eq) return idx; // found
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return m->cap; // full (should not happen with load factor control)
}

// ── Resize ────────────────────────────────────────────────────────────────────
int map_resize_(struct HashMap* m, unsigned long new_cap) {
    struct HashMap nm = map_alloc_table_(m->key_size, m->val_size, new_cap);
    if (nm.buckets == (struct MapEntry*)0) return 0;
    unsigned long i = 0UL;
    while (i < m->cap) {
        if (m->buckets[i].state == 1) {
            unsigned int h = m->buckets[i].hash;
            unsigned long idx = (unsigned long)(h & (unsigned int)(new_cap - 1UL));
            unsigned long j = 0UL;
            while (nm.buckets[idx].state == 1) {
                idx = (idx + 1UL) % new_cap;
                j = j + 1UL;
            }
            nm.buckets[idx] = m->buckets[i];
            nm.len = nm.len + 1UL;
        }
        i = i + 1UL;
    }
    unsafe { dealloc((void*)m->buckets); }
    m->buckets = nm.buckets;
    m->cap = new_cap;
    return 1;
}

// ── Core operations ───────────────────────────────────────────────────────────
int map_insert(struct HashMap* m, const void* key, const void* val) {
    // Resize if load factor > 0.75
    if (m->len * 4UL >= m->cap * 3UL) {
        if (!map_resize_(m, m->cap * 2UL)) return 0;
    }
    unsigned int h = map_hash_bytes_(key, m->key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0 || s == 2) {
            // Empty or tombstone — insert here
            unsafe {
                if (m->buckets[idx].key == (void*)0)
                    m->buckets[idx].key = alloc(m->key_size);
                if (m->buckets[idx].val == (void*)0)
                    m->buckets[idx].val = alloc(m->val_size);
                if (m->buckets[idx].key == (void*)0 || m->buckets[idx].val == (void*)0) return 0;
                safe_memcpy(m->buckets[idx].key, key, m->key_size);
                safe_memcpy(m->buckets[idx].val, val, m->val_size);
            }
            m->buckets[idx].hash = h;
            m->buckets[idx].state = 1;
            m->len = m->len + 1UL;
            return 1;
        }
        if (s == 1 && m->buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)m->buckets[idx].key, key, m->key_size) == 0;
            if (eq) {
                // Update value
                unsafe { safe_memcpy(m->buckets[idx].val, val, m->val_size); }
                return 1;
            }
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return 0;
}

void* map_get(struct HashMap* m, const void* key) {
    if (m->len == 0UL) return (void*)0;
    unsigned int h = map_hash_bytes_(key, m->key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) return (void*)0;
        if (s == 1 && m->buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)m->buckets[idx].key, key, m->key_size) == 0;
            if (eq) return m->buckets[idx].val;
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return (void*)0;
}

int map_contains(struct HashMap* m, const void* key) {
    return map_get(m, key) != (void*)0;
}

int map_remove(struct HashMap* m, const void* key) {
    if (m->len == 0UL) return 0;
    unsigned int h = map_hash_bytes_(key, m->key_size);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) return 0;
        if (s == 1 && m->buckets[idx].hash == h) {
            int eq = safe_memcmp((const void*)m->buckets[idx].key, key, m->key_size) == 0;
            if (eq) {
                m->buckets[idx].state = 2; // tombstone
                m->len = m->len - 1UL;
                return 1;
            }
        }
        idx = (idx + 1UL) % m->cap;
        i = i + 1UL;
    }
    return 0;
}

void map_foreach(struct HashMap* m, void* fn) {
    unsafe {
        void (*f)(const void*, void*) = (void (*)(const void*, void*))fn;
        unsigned long i = 0UL;
        while (i < m->cap) {
            if (m->buckets[i].state == 1)
                f((const void*)m->buckets[i].key, m->buckets[i].val);
            i = i + 1UL;
        }
    }
}

// ── String-keyed map ──────────────────────────────────────────────────────────
// String keys stored as pointer-sized values (const char* pointer)
struct HashMap str_map_new(unsigned long val_size) {
    // key_size = sizeof(char*) = 8
    return map_alloc_table_(8UL, val_size, 16UL);
}

int str_map_insert(struct HashMap* m, const char* key, const void* val) {
    if (m->len * 4UL >= m->cap * 3UL) {
        if (!map_resize_(m, m->cap * 2UL)) return 0;
    }
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0 || s == 2) {
            unsafe {
                // Duplicate the string and store pointer
                unsigned long klen = str_len(key) + 1UL;
                char* kcopy = (char*)alloc(klen);
                if (kcopy == (char*)0) return 0;
                safe_memcpy((void*)kcopy, (const void*)key, klen);
                if (s == 1 && m->buckets[idx].key != (void*)0)
                    dealloc(m->buckets[idx].key);
                m->buckets[idx].key = (void*)kcopy;
                if (m->buckets[idx].val == (void*)0)
                    m->buckets[idx].val = alloc(m->val_size);
                if (m->buckets[idx].val == (void*)0) return 0;
                safe_memcpy(m->buckets[idx].val, val, m->val_size);
            }
            m->buckets[idx].hash = h;
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
    if (m->len == 0UL) return (void*)0;
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) return (void*)0;
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
    if (m->len == 0UL) return 0;
    unsigned int h = map_hash_str_(key);
    unsigned long idx = (unsigned long)(h & (unsigned int)(m->cap - 1UL));
    unsigned long i = 0UL;
    while (i < m->cap) {
        int s = m->buckets[idx].state;
        if (s == 0) return 0;
        if (s == 1 && m->buckets[idx].hash == h) {
            unsafe {
                if (str_cmp((const char*)m->buckets[idx].key, key) == 0) {
                    dealloc(m->buckets[idx].key);
                    m->buckets[idx].key = (void*)0;
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

generic<T>
int map_insert_t(struct HashMap* m, const void* key, T val) {
    return map_insert(m, key, (const void*)&val);
}

generic<T>
T* map_get_t(struct HashMap* m, const void* key) {
    return (T*)map_get(m, key);
}
