// SafeC Standard Library — B-Tree Ordered Map
// Deterministic O(log n) ordered key-value store; keys are unsigned long.
// Uses a fixed-degree B-tree (order 4 → up to 7 keys per node).
// Freestanding-safe; all node allocation via caller-supplied allocator or heap.
#pragma once

#define BTREE_ORDER      4      // min keys per non-root node
#define BTREE_MAX_KEYS   7      // 2*ORDER - 1
#define BTREE_MAX_CHILD  8      // 2*ORDER

namespace std {

struct BTreeNode {
    unsigned long keys[BTREE_MAX_KEYS];
    void*         vals[BTREE_MAX_KEYS];
    unsigned long children[BTREE_MAX_CHILD]; // indices into node pool (0 = null)
    int           n;            // current number of keys
    int           leaf;         // 1 if leaf node
};

#define BTREE_POOL_SIZE  256    // max nodes in the static pool

struct BTree {
    struct BTreeNode pool[BTREE_POOL_SIZE];
    int              pool_used;
    unsigned long    root;      // index into pool (0 = empty tree)
    unsigned long    count;     // total key-value pairs

    // Insert or update key → val.  Returns 0 on success, -1 if pool full.
    int  insert(unsigned long key, void* val);

    // Lookup key.  Returns val pointer, or NULL if not found.
    void* get(unsigned long key) const;

    // Remove key.  Returns 1 if found and removed, 0 if not found.
    int  remove(unsigned long key);

    // Return number of entries.
    unsigned long len() const;

    // Return 1 if key exists.
    int  contains(unsigned long key) const;

    // In-order traversal.  Calls cb(key, val, user) for every entry.
    void foreach(void* cb, void* user) const;

    // Clear all entries.
    void clear();
};

// Initialise (or reset) an empty tree in place — matches the
// '<foo>_init(&stack Ctx ctx)' convention other pool-backed std/ structs
// use (see std/fs/tmpfs.h's tmpfs_init, std/fs/fat.h's fat_init), rather
// than a by-value 'btree_new()' constructor: only the three scalar fields
// above are touched. 'pool' is intentionally left untouched — nodes are
// claimed from it lazily as 'pool_used' grows, so zeroing all ~47KB of it
// up front would be pure waste, and returning a fresh 'struct BTree' by
// value would copy that whole pool on every call.
void btree_init(&stack BTree t);

// Typed generic wrappers (value stored as pointer; caller manages lifetime).
generic<T> int  btree_insert(&stack BTree t, unsigned long key, T* val);
generic<T> T*   btree_get(const &stack BTree t, unsigned long key);

} // namespace std
