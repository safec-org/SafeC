#pragma once
// SafeC Standard Library — HashMap<K,V> / StrMap<V> (open addressing, linear probing)
//
//   struct HashMap<int, long long> m = std::map_new();   // K,V inferred from the target
//   m.insert(7, 49LL);
//   match (m.get(7)) { case null: ... case some(v): *v = 50LL; }
//   ?long long gone = m.remove(7);
//   m.free();
//
//   struct StrMap<int> sm = std::str_map_new();          // C-string keys (copied)
//   sm.insert("one", 1);
//
// HashMap keys are hashed and compared by their bytes, so K must be a type
// whose value is fully described by its bytes: integers, enums, pointers,
// and structs without padding. Use StrMap for text keys.
//
// get() hands out '?&V' — a reference into the table, valid until the next
// insert()/remove()/clear()/free() (any of which may move or drop entries).
// The table resizes at a 0.75 load factor; tombstones count toward it, so a
// remove-heavy workload rehashes instead of degrading every probe to O(cap).
#include <std/mem.h>
#include <std/str.h>
#include <std/collections/string.h>

namespace std {

// Slot states (one byte per slot, kept in their own array so probing touches
// as little memory as possible).
#define MAP_EMPTY_ 0
#define MAP_FULL_  1
#define MAP_TOMB_  2

unsigned int  map_hash_str_(const char* s);    // map.sc

// Keys of 1/2/4/8 bytes (every integer, enum, pointer and small padding-
// free struct) are read as one little-endian word — assembled from bytes,
// so LLVM emits a single unaligned-safe load — then hashed with murmur3's
// 64-bit finalizer and compared with one integer compare. sizeof(K) is a
// constant in each instance, so 'if const' picks the branch at compile time. Other sizes fall
// back to FNV-1a over the bytes and a byte-wise compare.
generic<K>
unsigned long long map_key_word_(const &K key) {
    unsigned long long w = 0ULL;
    unsafe {
        const unsigned char* p = (const unsigned char*)key;
        unsigned long i = 0UL;
        while (i < sizeof(K)) { w = w | ((unsigned long long)p[i] << (i * 8UL)); i = i + 1UL; }
    }
    return w;
}

generic<K>
unsigned int map_key_hash_(const &K key) {
    if const (sizeof(K) == 1UL || sizeof(K) == 2UL || sizeof(K) == 4UL || sizeof(K) == 8UL) {
        unsigned long long x = map_key_word_(key);
        x = x ^ (x >> 33); x = x * 0xff51afd7ed558ccdULL;
        x = x ^ (x >> 33); x = x * 0xc4ceb9fe1a85ec53ULL;
        x = x ^ (x >> 33);
        return (unsigned int)x;
    }
    unsigned int h = 2166136261U;
    unsafe {
        const unsigned char* p = (const unsigned char*)key;
        unsigned long i = 0UL;
        while (i < sizeof(K)) { h = (h ^ (unsigned int)p[i]) * 16777619U; i = i + 1UL; }
    }
    h = h ^ (h >> 16); h = h * 2246822507U;
    h = h ^ (h >> 13); h = h * 3266489909U;
    return h ^ (h >> 16);
}

generic<K>
int map_key_eq_(const &K a, const &K b) {
    if const (sizeof(K) == 1UL || sizeof(K) == 2UL || sizeof(K) == 4UL || sizeof(K) == 8UL) return map_key_word_(a) == map_key_word_(b);
    unsafe {
        const unsigned char* pa = (const unsigned char*)a;
        const unsigned char* pb = (const unsigned char*)b;
        unsigned long i = 0UL;
        while (i < sizeof(K)) { if (pa[i] != pb[i]) return 0; i = i + 1UL; }
    }
    return 1;
}

unsigned long map_round_cap_(unsigned long n); // map.sc: smallest power of two >= max(16, n)

// ══ HashMap<K,V> ══════════════════════════════════════════════════════════════

generic<K, V>
struct HashMap {
    ?&heap K             keys;
    ?&heap V             vals;
    ?&heap unsigned int  hashes;
    ?&heap unsigned char states;     // MAP_EMPTY_ / MAP_FULL_ / MAP_TOMB_
    unsigned long        cap;        // 0 or a power of two
    unsigned long        len;        // live entries
    unsigned long        tombstones; // removed slots not yet reclaimed

    // ── Core operations ──────────────────────────────────────────────────────
    int           insert(K key, V val);       // insert or overwrite; 0 on OOM
    ?&V           get(K key) const;           // null if missing
    int           contains(K key) const;
    ?V            remove(K key);              // the removed value, or null
    unsigned long length() const;
    int           is_empty() const;
    void          clear();                     // drop every entry, keep the table
    void          foreach(fn void(const &K, &V) func);

    // ── Lifecycle ────────────────────────────────────────────────────────────
    int           reserve(unsigned long n);   // room for n entries without resizing
    void          free();

    // ── Internal ────────────────────────────────────────────────────────────
    long long     find_(const &K key, unsigned int h) const;
    int           rehash_(unsigned long new_cap);
};

generic<K, V>
struct HashMap<K, V> map_new() {
    struct HashMap<K, V> m;
    m.keys = null; m.vals = null; m.hashes = null; m.states = null;
    m.cap = 0UL; m.len = 0UL; m.tombstones = 0UL;
    return m;
}

generic<K, V>
struct HashMap<K, V> map_with_cap(unsigned long cap) {
    struct HashMap<K, V> m = map_new();
    m.reserve(cap);
    return m;
}

generic<K, V>
unsigned long HashMap::length() const { return self.len; }

generic<K, V>
int HashMap::is_empty() const { return self.len == 0UL; }

// The lookup and insert paths are inlined into callers (as a C map's
// static functions would be); the rehash stays an out-of-line call.
generic<K, V>
long long HashMap::find_(const &K key, unsigned int h) const __attribute__((always_inline)) {
    if (self.len == 0UL) return -1LL;
    unsigned long cap = self.cap;
    unsigned long mask = cap - 1UL;
    unsigned long idx = (unsigned long)h & mask;
    unsigned long n = 0UL;
    unsafe {
        const unsigned char* st = (const unsigned char*)self.states;
        const unsigned int* hs = (const unsigned int*)self.hashes;
        const K* ks = (const K*)self.keys;
        while (n < cap) {
            unsigned char s = st[idx];
            if (s == (unsigned char)MAP_EMPTY_) return -1LL;
            if (s == (unsigned char)MAP_FULL_ && hs[idx] == h && map_key_eq_(&ks[idx], key))
                return (long long)idx;
            idx = (idx + 1UL) & mask;
            n = n + 1UL;
        }
    }
    return -1LL;
}

// Moves every live entry into fresh arrays of 'new_cap' slots (a power of
// two), dropping tombstones. Leaves the map untouched on OOM.
generic<K, V>
int HashMap::rehash_(unsigned long new_cap) __attribute__((noinline, cold)) {
    ?&heap K             nk = null;
    ?&heap V             nv = null;
    ?&heap unsigned int  nh = null;
    ?&heap unsigned char ns = null;
    unsafe {
        nk = (?&heap K)alloc_raw_(checked_mul_size(new_cap, sizeof(K)));
        nv = (?&heap V)alloc_raw_(checked_mul_size(new_cap, sizeof(V)));
        nh = (?&heap unsigned int)alloc_raw_(checked_mul_size(new_cap, sizeof(unsigned int)));
        ns = (?&heap unsigned char)alloc_zeroed_raw_(new_cap);
    }
    if (nk.is_null() || nv.is_null() || nh.is_null() || ns.is_null()) {
        unsafe { dealloc_raw_((void*)nk); dealloc_raw_((void*)nv); dealloc_raw_((void*)nh); dealloc_raw_((void*)ns); }
        return 0;
    }
    unsigned long mask = new_cap - 1UL;
    unsafe {
        // The old table in locals: the byte stores into 'ns' may alias
        // anything, so fields read through self would be reloaded per slot.
        const K* ok = (const K*)self.keys;
        const V* ov = (const V*)self.vals;
        const unsigned int* oh = (const unsigned int*)self.hashes;
        const unsigned char* os = (const unsigned char*)self.states;
        K* tk = (K*)nk;
        V* tv = (V*)nv;
        unsigned int* th = (unsigned int*)nh;
        unsigned char* ts = (unsigned char*)ns;
        unsigned long oc = self.cap;
        unsigned long i = 0UL;
        while (i < oc) {
            if (os[i] == (unsigned char)MAP_FULL_) {
                unsigned int h = oh[i];
                unsigned long idx = (unsigned long)h & mask;
                while (ts[idx] != (unsigned char)MAP_EMPTY_) { idx = (idx + 1UL) & mask; }
                tk[idx] = ok[i];
                tv[idx] = ov[i];
                th[idx] = h;
                ts[idx] = (unsigned char)MAP_FULL_;
            }
            i = i + 1UL;
        }
        dealloc_raw_((void*)self.keys); dealloc_raw_((void*)self.vals);
        dealloc_raw_((void*)self.hashes); dealloc_raw_((void*)self.states);
    }
    self.keys = nk; self.vals = nv; self.hashes = nh; self.states = ns;
    self.cap = new_cap;
    self.tombstones = 0UL;
    return 1;
}

generic<K, V>
int HashMap::reserve(unsigned long n) {
    // keep len <= 3/4 cap
    unsigned long want = map_round_cap_(checked_add_size(n, n / 3UL + 1UL));
    if (want <= self.cap) return 1;
    return self.rehash_(want);
}

generic<K, V>
int HashMap::insert(K key, V val) __attribute__((always_inline)) {
    if ((self.len + self.tombstones + 1UL) * 4UL > self.cap * 3UL) {
        // Mostly tombstones: rehash in place; otherwise grow.
        unsigned long next = self.cap == 0UL ? 16UL
                           : ((self.len + 1UL) * 2UL > self.cap ? self.cap * 2UL : self.cap);
        if (!self.rehash_(next)) return 0;
    }
    unsigned int h = map_key_hash_(&key);
    unsigned long mask = self.cap - 1UL;
    unsigned long idx = (unsigned long)h & mask;
    long long tomb = -1LL;
    unsafe {
        const unsigned char* st = (const unsigned char*)self.states;
        const unsigned int* hs = (const unsigned int*)self.hashes;
        K* ks = (K*)self.keys;
        while (1) {
            unsigned char s = st[idx];
            if (s == (unsigned char)MAP_FULL_) {
                if (hs[idx] == h && map_key_eq_(&ks[idx], &key)) {
                    self.vals[idx] = val;
                    return 1;
                }
            } else if (s == (unsigned char)MAP_EMPTY_) {
                break;
            } else if (tomb < 0LL) {
                tomb = (long long)idx;
            }
            idx = (idx + 1UL) & mask;
        }
        if (tomb >= 0LL) {
            idx = (unsigned long)tomb;
            self.tombstones = self.tombstones - 1UL;
        }
        ks[idx] = key;
        self.vals[idx] = val;
        self.hashes[idx] = h;
        self.states[idx] = (unsigned char)MAP_FULL_;
    }
    self.len = self.len + 1UL;
    return 1;
}

generic<K, V>
?&V HashMap::get(K key) const __attribute__((always_inline)) {
    long long i = self.find_(&key, map_key_hash_(&key));
    if (i < 0LL) return null;
    unsafe { return (?&V)&self.vals[(unsigned long)i]; }
}

generic<K, V>
int HashMap::contains(K key) const __attribute__((always_inline)) {
    return self.find_(&key, map_key_hash_(&key)) >= 0LL;
}

generic<K, V>
?V HashMap::remove(K key) {
    long long i = self.find_(&key, map_key_hash_(&key));
    if (i < 0LL) return null;
    unsigned long idx = (unsigned long)i;
    self.len = self.len - 1UL;
    self.tombstones = self.tombstones + 1UL;
    unsafe {
        self.states[idx] = (unsigned char)MAP_TOMB_;
        return self.vals[idx];
    }
}

generic<K, V>
void HashMap::clear() {
    if (self.cap != 0UL) {
        unsafe { safe_memset_raw_((void*)self.states, 0, self.cap); }
    }
    self.len = 0UL;
    self.tombstones = 0UL;
}

generic<K, V>
void HashMap::foreach(fn void(const &K, &V) func) {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.cap) {
            if (self.states[i] == (unsigned char)MAP_FULL_) func(&self.keys[i], &self.vals[i]);
            i = i + 1UL;
        }
    }
}

generic<K, V>
void HashMap::free() {
    unsafe {
        dealloc_raw_((void*)self.keys); dealloc_raw_((void*)self.vals);
        dealloc_raw_((void*)self.hashes); dealloc_raw_((void*)self.states);
    }
    self.keys = null; self.vals = null; self.hashes = null; self.states = null;
    self.cap = 0UL; self.len = 0UL; self.tombstones = 0UL;
}

// ══ StrMap<V> — C-string keys, copied into the map ═══════════════════════════

generic<V>
struct StrMap {
    ?&heap struct String keys;       // owned copies; valid while the slot is MAP_FULL_
    ?&heap V             vals;
    ?&heap unsigned int  hashes;
    ?&heap unsigned char states;
    unsigned long        cap;
    unsigned long        len;
    unsigned long        tombstones;

    int           insert(const &char key, V val);   // copies key; 0 on OOM
    ?&V           get(const &char key) const;
    int           contains(const &char key) const;
    ?V            remove(const &char key);
    unsigned long length() const;
    int           is_empty() const;
    void          clear();
    void          foreach(fn void(const &char, &V) func);
    void          free();

    long long     find_(const char* key, unsigned int h) const;
    int           rehash_(unsigned long new_cap);
};

generic<V>
struct StrMap<V> str_map_new() {
    struct StrMap<V> m;
    m.keys = null; m.vals = null; m.hashes = null; m.states = null;
    m.cap = 0UL; m.len = 0UL; m.tombstones = 0UL;
    return m;
}

generic<V>
unsigned long StrMap::length() const { return self.len; }

generic<V>
int StrMap::is_empty() const { return self.len == 0UL; }

generic<V>
long long StrMap::find_(const char* key, unsigned int h) const {
    if (self.len == 0UL) return -1LL;
    unsigned long mask = self.cap - 1UL;
    unsigned long idx = (unsigned long)h & mask;
    unsigned long n = 0UL;
    unsafe {
        while (n < self.cap) {
            unsigned char s = self.states[idx];
            if (s == (unsigned char)MAP_EMPTY_) return -1LL;
            if (s == (unsigned char)MAP_FULL_ && self.hashes[idx] == h
                && str_cmp(self.keys[idx].as_ptr(), key) == 0)
                return (long long)idx;
            idx = (idx + 1UL) & mask;
            n = n + 1UL;
        }
    }
    return -1LL;
}

generic<V>
int StrMap::rehash_(unsigned long new_cap) {
    ?&heap struct String nk = null;
    ?&heap V             nv = null;
    ?&heap unsigned int  nh = null;
    ?&heap unsigned char ns = null;
    unsafe {
        nk = (?&heap struct String)alloc_raw_(checked_mul_size(new_cap, sizeof(struct String)));
        nv = (?&heap V)alloc_raw_(checked_mul_size(new_cap, sizeof(V)));
        nh = (?&heap unsigned int)alloc_raw_(checked_mul_size(new_cap, sizeof(unsigned int)));
        ns = (?&heap unsigned char)alloc_zeroed_raw_(new_cap);
    }
    if (nk.is_null() || nv.is_null() || nh.is_null() || ns.is_null()) {
        unsafe { dealloc_raw_((void*)nk); dealloc_raw_((void*)nv); dealloc_raw_((void*)nh); dealloc_raw_((void*)ns); }
        return 0;
    }
    unsigned long mask = new_cap - 1UL;
    unsafe {
        unsigned long i = 0UL;
        while (i < self.cap) {
            if (self.states[i] == (unsigned char)MAP_FULL_) {
                unsigned int h = self.hashes[i];
                unsigned long idx = (unsigned long)h & mask;
                while (ns[idx] != (unsigned char)MAP_EMPTY_) { idx = (idx + 1UL) & mask; }
                nk[idx] = self.keys[i];      // moves the String; old slot is dropped
                nv[idx] = self.vals[i];
                nh[idx] = h;
                ns[idx] = (unsigned char)MAP_FULL_;
            }
            i = i + 1UL;
        }
        dealloc_raw_((void*)self.keys); dealloc_raw_((void*)self.vals);
        dealloc_raw_((void*)self.hashes); dealloc_raw_((void*)self.states);
    }
    self.keys = nk; self.vals = nv; self.hashes = nh; self.states = ns;
    self.cap = new_cap;
    self.tombstones = 0UL;
    return 1;
}

generic<V>
int StrMap::insert(const &char key, V val) {
    if ((self.len + self.tombstones + 1UL) * 4UL > self.cap * 3UL) {
        unsigned long next = self.cap == 0UL ? 16UL
                           : ((self.len + 1UL) * 2UL > self.cap ? self.cap * 2UL : self.cap);
        if (!self.rehash_(next)) return 0;
    }
    unsigned int h = map_hash_str_(key);
    unsigned long mask = self.cap - 1UL;
    unsigned long idx = (unsigned long)h & mask;
    long long tomb = -1LL;
    unsafe {
        while (1) {
            unsigned char s = self.states[idx];
            if (s == (unsigned char)MAP_EMPTY_) break;
            if (s == (unsigned char)MAP_TOMB_) {
                if (tomb < 0LL) tomb = (long long)idx;
            } else if (self.hashes[idx] == h && str_cmp(self.keys[idx].as_ptr(), key) == 0) {
                self.vals[idx] = val;
                return 1;
            }
            idx = (idx + 1UL) & mask;
        }
        if (tomb >= 0LL) {
            idx = (unsigned long)tomb;
            self.tombstones = self.tombstones - 1UL;
        }
        self.keys[idx] = string_from(key);
        self.vals[idx] = val;
        self.hashes[idx] = h;
        self.states[idx] = (unsigned char)MAP_FULL_;
    }
    self.len = self.len + 1UL;
    return 1;
}

generic<V>
?&V StrMap::get(const &char key) const {
    long long i = self.find_(key, map_hash_str_(key));
    if (i < 0LL) return null;
    unsafe { return (?&V)&self.vals[(unsigned long)i]; }
}

generic<V>
int StrMap::contains(const &char key) const {
    return self.find_(key, map_hash_str_(key)) >= 0LL;
}

generic<V>
?V StrMap::remove(const &char key) {
    long long i = self.find_(key, map_hash_str_(key));
    if (i < 0LL) return null;
    unsigned long idx = (unsigned long)i;
    self.len = self.len - 1UL;
    self.tombstones = self.tombstones + 1UL;
    unsafe {
        self.keys[idx].free();
        self.states[idx] = (unsigned char)MAP_TOMB_;
        return self.vals[idx];
    }
}

generic<V>
void StrMap::clear() {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.cap) {
            if (self.states[i] == (unsigned char)MAP_FULL_) self.keys[i].free();
            self.states[i] = (unsigned char)MAP_EMPTY_;
            i = i + 1UL;
        }
    }
    self.len = 0UL;
    self.tombstones = 0UL;
}

generic<V>
void StrMap::foreach(fn void(const &char, &V) func) {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.cap) {
            if (self.states[i] == (unsigned char)MAP_FULL_) func(self.keys[i].as_ptr(), &self.vals[i]);
            i = i + 1UL;
        }
    }
}

generic<V>
void StrMap::free() {
    self.clear();
    unsafe {
        dealloc_raw_((void*)self.keys); dealloc_raw_((void*)self.vals);
        dealloc_raw_((void*)self.hashes); dealloc_raw_((void*)self.states);
    }
    self.keys = null; self.vals = null; self.hashes = null; self.states = null;
    self.cap = 0UL;
}

} // namespace std
