#pragma once
// SafeC Standard Library — BST (unbalanced binary search tree)
// cmp_fn: int(*)(const void* a, const void* b) → <0, 0, >0
// Keys are heap-copies; values are heap-copies.

namespace std {

struct BSTNode {
    void*        key;   // generic-erasure (element type is caller-parametric — see gui_widget.h's field comment for the same shape)
    void*        val;
    ?&heap BSTNode left;  // empty (null) for a leaf's missing child
    ?&heap BSTNode right;
};

struct BST {
    ?&heap BSTNode root;  // empty (null) for an empty tree; heap-owned by this BST
    unsigned long   key_size;
    unsigned long   val_size;
    void*           cmp_fn;   // int(*)(const void*, const void*)
    unsigned long   len;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct BST bst_new(unsigned long key_size, unsigned long val_size, void* cmp_fn);
void       bst_free(&BST t);

// ── Core operations ───────────────────────────────────────────────────────────
int   bst_insert(&BST t, const void* key, const void* val);
void* bst_get(&BST t, const void* key);   // NULL if not found
int   bst_contains(&BST t, const void* key);
int   bst_remove(&BST t, const void* key);
unsigned long bst_len(&BST t);
int   bst_is_empty(&BST t);
void  bst_clear(&BST t);

// ── Min / max ─────────────────────────────────────────────────────────────────
void* bst_min_key(&BST t);  // NULL if empty
void* bst_max_key(&BST t);  // NULL if empty

// ── Traversal (in-order = sorted ascending) ────────────────────────────────────
void bst_foreach_inorder(&BST t, void* func); // func: void(*)(const void* key, void* val)

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int bst_insert_t(&BST t, const void* key, T val);

generic<T>
T* bst_get_t(&BST t, const void* key);

// ── Built-in comparators (pass as cmp_fn) ─────────────────────────────────────
int bst_cmp_int(const void* a, const void* b);          // int keys
int bst_cmp_ll(const void* a, const void* b);           // long long keys
int bst_cmp_str(const void* a, const void* b);          // const char* keys (pointer inside)
int bst_cmp_uint(const void* a, const void* b);         // unsigned int keys

} // namespace std
