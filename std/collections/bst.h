#pragma once
// SafeC Standard Library — BST (unbalanced binary search tree)
// cmp_fn: int(*)(const void* a, const void* b) → <0, 0, >0
// Keys are heap-copies; values are heap-copies.

struct BSTNode {
    void*            key;
    void*            val;
    struct BSTNode*  left;
    struct BSTNode*  right;
};

struct BST {
    struct BSTNode* root;
    unsigned long   key_size;
    unsigned long   val_size;
    void*           cmp_fn;   // int(*)(const void*, const void*)
    unsigned long   len;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct BST bst_new(unsigned long key_size, unsigned long val_size, void* cmp_fn);
void       bst_free(struct BST* t);

// ── Core operations ───────────────────────────────────────────────────────────
int   bst_insert(struct BST* t, const void* key, const void* val);
void* bst_get(struct BST* t, const void* key);   // NULL if not found
int   bst_contains(struct BST* t, const void* key);
int   bst_remove(struct BST* t, const void* key);
unsigned long bst_len(struct BST* t);
int   bst_is_empty(struct BST* t);
void  bst_clear(struct BST* t);

// ── Min / max ─────────────────────────────────────────────────────────────────
void* bst_min_key(struct BST* t);  // NULL if empty
void* bst_max_key(struct BST* t);  // NULL if empty

// ── Traversal (in-order = sorted ascending) ────────────────────────────────────
void bst_foreach_inorder(struct BST* t, void* fn); // fn: void(*)(const void* key, void* val)

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int bst_insert_t(struct BST* t, const void* key, T val);

generic<T>
T* bst_get_t(struct BST* t, const void* key);

// ── Built-in comparators (pass as cmp_fn) ─────────────────────────────────────
int bst_cmp_int(const void* a, const void* b);          // int keys
int bst_cmp_ll(const void* a, const void* b);           // long long keys
int bst_cmp_str(const void* a, const void* b);          // const char* keys (pointer inside)
int bst_cmp_uint(const void* a, const void* b);         // unsigned int keys
