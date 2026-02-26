#pragma once
// SafeC Standard Library — HashMap (open addressing, linear probing)
// Keys are compared byte-by-byte (memcmp). Use str_map_* for C-string keys.
// Load factor threshold: 0.75 — resizes automatically.

struct MapEntry {
    void*        key;
    void*        val;
    unsigned int hash;
    int          state; // 0=empty, 1=occupied, 2=tombstone
};

struct HashMap {
    struct MapEntry* buckets;  // heap-allocated bucket array
    unsigned long    cap;      // must be power of 2
    unsigned long    len;      // live entries
    unsigned long    key_size;
    unsigned long    val_size;

    // ── Core operations ──────────────────────────────────────────────────────
    int           insert(const void* key, const void* val);
    void*         get(const void* key) const;    // NULL if missing
    int           contains(const void* key) const;
    int           remove(const void* key);
    unsigned long length() const;
    int           is_empty() const;
    void          clear();
    void          foreach(void* fn); // fn: void(*)(const void* key, void* val)

    // ── Lifecycle ────────────────────────────────────────────────────────────
    void          free();

    // ── Internal ────────────────────────────────────────────────────────────
    int           resize_(unsigned long new_cap);
    void          free_entries_();
};

// ── Constructor free functions ────────────────────────────────────────────────
struct HashMap map_new(unsigned long key_size, unsigned long val_size);
struct HashMap map_with_cap(unsigned long key_size, unsigned long val_size, unsigned long cap);

// ── String-keyed convenience (keys are const char*) ───────────────────────────
struct HashMap str_map_new(unsigned long val_size);
int   str_map_insert(struct HashMap* m, const char* key, const void* val);
void* str_map_get(struct HashMap* m, const char* key);
int   str_map_contains(struct HashMap* m, const char* key);
int   str_map_remove(struct HashMap* m, const char* key);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int map_insert_t(&stack HashMap m, const void* key, T val);

generic<T>
T* map_get_t(&stack HashMap m, const void* key);
