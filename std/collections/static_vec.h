// SafeC Standard Library — StaticVec<T> / StaticMap<V>: zero-heap collections
//
// Fixed-capacity containers over storage the caller provides — a static
// array, a stack array, an arena block — so nothing is ever allocated and
// the capacity is whatever the backing array's (compile-time) size is.
// Freestanding-safe.
//
//   static int g_buf[32];
//   struct StaticVec<int> v = std::StaticVec::over(&g_buf[0], 32UL);
//   v.push(42);
//   ?int top = v.pop();
//
//   static struct StaticMapSlot<int> g_slots[64];
//   struct StaticMap<int> m = std::StaticMap::over(&g_slots[0], 64UL);
//   m.insert(5UL, 100);
//   int x = m.get(5UL).default(0);
//
// The container holds a region-less reference to the storage: keep the
// backing array alive at least as long as the container.
#pragma once

namespace std {

// ── StaticVec<T> ──────────────────────────────────────────────────────────────

generic<T>
struct StaticVec {
    &T            data;   // caller's storage, 'cap' elements
    unsigned long len;
    unsigned long cap;

    static struct StaticVec<T> over(&T storage, unsigned long cap);  // empty, over cap elements

    int           push(T val);                  // 0 if full
    ?T            pop();                        // null if empty
    ?&T           top() const;
    ?&T           get(unsigned long idx) const; // null if out of range
    unsigned long length() const;
    int           is_empty() const;
    int           is_full() const;
    void          clear();
};

generic<T>
struct StaticVec<T> StaticVec::over(&T storage, unsigned long cap) {
    struct StaticVec<T> v;
    v.data = storage;
    v.len = 0UL;
    v.cap = cap;
    return v;
}

generic<T>
int StaticVec::push(T val) {
    if (self.len >= self.cap) return 0;
    unsafe { self.data[self.len] = val; }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
?T StaticVec::pop() {
    if (self.len == 0UL) return null;
    self.len = self.len - 1UL;
    unsafe { return self.data[self.len]; }
}

generic<T>
?&T StaticVec::top() const {
    if (self.len == 0UL) return null;
    unsafe { return (?&T)&self.data[self.len - 1UL]; }
}

generic<T>
?&T StaticVec::get(unsigned long idx) const {
    if (idx >= self.len) return null;
    unsafe { return (?&T)&self.data[idx]; }
}

generic<T>
unsigned long StaticVec::length() const { return self.len; }

generic<T>
int StaticVec::is_empty() const { return self.len == 0UL; }

generic<T>
int StaticVec::is_full() const { return self.len >= self.cap; }

generic<T>
void StaticVec::clear() { self.len = 0UL; }

// ── StaticMap<V> (unsigned long keys, open addressing) ────────────────────────

generic<V>
struct StaticMapSlot {
    unsigned long key;
    V             val;
    int           used;
};

generic<V>
struct StaticMap {
    &StaticMapSlot<V> slots;   // caller's storage, 'cap' slots
    unsigned long     cap;
    unsigned long     count;

    static struct StaticMap<V> over(&StaticMapSlot<V> storage, unsigned long cap);  // empty, over cap slots

    int           insert(unsigned long key, V val);   // insert or update; 0 if full
    ?&V           get(unsigned long key) const;       // null if missing
    int           contains(unsigned long key) const;
    unsigned long length() const;
    void          clear();
};

generic<V>
struct StaticMap<V> StaticMap::over(&StaticMapSlot<V> storage, unsigned long cap) {
    struct StaticMap<V> m;
    m.slots = storage;
    m.cap = cap;
    m.count = 0UL;
    m.clear();
    return m;
}

generic<V>
int StaticMap::insert(unsigned long key, V val) {
    unsigned long i = 0UL;
    while (i < self.cap) {
        unsigned long s = (key + i) % self.cap;
        unsafe {
            if (!self.slots[s].used || self.slots[s].key == key) {
                if (!self.slots[s].used) { self.slots[s].used = 1; self.count = self.count + 1UL; }
                self.slots[s].key = key;
                self.slots[s].val = val;
                return 1;
            }
        }
        i = i + 1UL;
    }
    return 0;
}

generic<V>
?&V StaticMap::get(unsigned long key) const {
    unsigned long i = 0UL;
    while (i < self.cap) {
        unsigned long s = (key + i) % self.cap;
        unsafe {
            if (!self.slots[s].used) return null;
            if (self.slots[s].key == key) return (?&V)&self.slots[s].val;
        }
        i = i + 1UL;
    }
    return null;
}

generic<V>
int StaticMap::contains(unsigned long key) const { return !self.get(key).is_null(); }

generic<V>
unsigned long StaticMap::length() const { return self.count; }

generic<V>
void StaticMap::clear() {
    unsafe {
        unsigned long i = 0UL;
        while (i < self.cap) { self.slots[i].used = 0; i = i + 1UL; }
    }
    self.count = 0UL;
}

} // namespace std
