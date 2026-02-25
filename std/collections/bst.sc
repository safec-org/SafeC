// SafeC Standard Library — BST implementation
#include "bst.h"
#include "../mem.h"
#include "../str.h"

// ── Built-in comparators ──────────────────────────────────────────────────────
int bst_cmp_int(const void* a, const void* b) {
    unsafe {
        int ia = *(int*)a;
        int ib = *(int*)b;
        return ia < ib ? -1 : (ia > ib ? 1 : 0);
    }
}
int bst_cmp_ll(const void* a, const void* b) {
    unsafe {
        long long la = *(long long*)a;
        long long lb = *(long long*)b;
        return la < lb ? -1 : (la > lb ? 1 : 0);
    }
}
int bst_cmp_str(const void* a, const void* b) {
    unsafe { return str_cmp(*(const char**)a, *(const char**)b); }
}
int bst_cmp_uint(const void* a, const void* b) {
    unsafe {
        unsigned int ua = *(unsigned int*)a;
        unsigned int ub = *(unsigned int*)b;
        return ua < ub ? -1 : (ua > ub ? 1 : 0);
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────
struct BSTNode* bst_alloc_node_(unsigned long key_size, unsigned long val_size,
                                 const void* key, const void* val) {
    unsafe {
        struct BSTNode* n = (struct BSTNode*)alloc(sizeof(struct BSTNode));
        if (n == (struct BSTNode*)0) return (struct BSTNode*)0;
        n->key = alloc(key_size);
        n->val = alloc(val_size);
        if (n->key == (void*)0 || n->val == (void*)0) {
            if (n->key != (void*)0) dealloc(n->key);
            if (n->val != (void*)0) dealloc(n->val);
            dealloc((void*)n);
            return (struct BSTNode*)0;
        }
        safe_memcpy(n->key, key, key_size);
        safe_memcpy(n->val, val, val_size);
        n->left  = (struct BSTNode*)0;
        n->right = (struct BSTNode*)0;
        return n;
    }
}

void bst_free_node_(struct BSTNode* n) {
    if (n == (struct BSTNode*)0) return;
    bst_free_node_(n->left);
    bst_free_node_(n->right);
    unsafe {
        dealloc(n->key);
        dealloc(n->val);
        dealloc((void*)n);
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct BST bst_new(unsigned long key_size, unsigned long val_size, void* cmp_fn) {
    struct BST t;
    t.root = (struct BSTNode*)0;
    t.key_size = key_size;
    t.val_size = val_size;
    t.cmp_fn = cmp_fn;
    t.len = 0UL;
    return t;
}

void bst_clear(struct BST* t) {
    bst_free_node_(t->root);
    t->root = (struct BSTNode*)0;
    t->len = 0UL;
}

void bst_free(struct BST* t) { bst_clear(t); }

unsigned long bst_len(struct BST* t)    { return t->len; }
int           bst_is_empty(struct BST* t) { return t->len == 0UL; }

// ── Insert (recursive via stack simulation using while) ───────────────────────
int bst_insert(struct BST* t, const void* key, const void* val) {
    unsafe {
        int (*cmp)(const void*, const void*) = (int (*)(const void*, const void*))t->cmp_fn;
        if (t->root == (struct BSTNode*)0) {
            t->root = bst_alloc_node_(t->key_size, t->val_size, key, val);
            if (t->root == (struct BSTNode*)0) return 0;
            t->len = t->len + 1UL;
            return 1;
        }
        struct BSTNode* cur = t->root;
        while (1) {
            int c = cmp(key, (const void*)cur->key);
            if (c == 0) {
                // Update value
                safe_memcpy(cur->val, val, t->val_size);
                return 1;
            } else if (c < 0) {
                if (cur->left == (struct BSTNode*)0) {
                    cur->left = bst_alloc_node_(t->key_size, t->val_size, key, val);
                    if (cur->left == (struct BSTNode*)0) return 0;
                    t->len = t->len + 1UL;
                    return 1;
                }
                cur = cur->left;
            } else {
                if (cur->right == (struct BSTNode*)0) {
                    cur->right = bst_alloc_node_(t->key_size, t->val_size, key, val);
                    if (cur->right == (struct BSTNode*)0) return 0;
                    t->len = t->len + 1UL;
                    return 1;
                }
                cur = cur->right;
            }
        }
    }
    return 0;
}

// ── Get ───────────────────────────────────────────────────────────────────────
void* bst_get(struct BST* t, const void* key) {
    unsafe {
        int (*cmp)(const void*, const void*) = (int (*)(const void*, const void*))t->cmp_fn;
        struct BSTNode* cur = t->root;
        while (cur != (struct BSTNode*)0) {
            int c = cmp(key, (const void*)cur->key);
            if (c == 0) return cur->val;
            cur = c < 0 ? cur->left : cur->right;
        }
    }
    return (void*)0;
}

int bst_contains(struct BST* t, const void* key) {
    return bst_get(t, key) != (void*)0;
}

// ── Min / max ─────────────────────────────────────────────────────────────────
void* bst_min_key(struct BST* t) {
    if (t->root == (struct BSTNode*)0) return (void*)0;
    struct BSTNode* n = t->root;
    while (n->left != (struct BSTNode*)0) n = n->left;
    return n->key;
}
void* bst_max_key(struct BST* t) {
    if (t->root == (struct BSTNode*)0) return (void*)0;
    struct BSTNode* n = t->root;
    while (n->right != (struct BSTNode*)0) n = n->right;
    return n->key;
}

// ── Remove (iterative) ────────────────────────────────────────────────────────
int bst_remove(struct BST* t, const void* key) {
    unsafe {
        int (*cmp)(const void*, const void*) = (int (*)(const void*, const void*))t->cmp_fn;
        struct BSTNode* parent = (struct BSTNode*)0;
        struct BSTNode* cur    = t->root;
        int went_left = 0;

        while (cur != (struct BSTNode*)0) {
            int c = cmp(key, (const void*)cur->key);
            if (c == 0) break;
            parent = cur;
            if (c < 0) { cur = cur->left;  went_left = 1; }
            else        { cur = cur->right; went_left = 0; }
        }
        if (cur == (struct BSTNode*)0) return 0; // not found

        struct BSTNode* replacement = (struct BSTNode*)0;
        if (cur->left == (struct BSTNode*)0) {
            replacement = cur->right;
        } else if (cur->right == (struct BSTNode*)0) {
            replacement = cur->left;
        } else {
            // Two children: replace with in-order successor (leftmost in right subtree)
            struct BSTNode* succ_parent = cur;
            struct BSTNode* succ = cur->right;
            while (succ->left != (struct BSTNode*)0) {
                succ_parent = succ;
                succ = succ->left;
            }
            // Copy successor key/val into cur
            safe_memcpy(cur->key, (const void*)succ->key, t->key_size);
            safe_memcpy(cur->val, (const void*)succ->val, t->val_size);
            // Remove successor node
            if (succ_parent == cur) succ_parent->right = succ->right;
            else                    succ_parent->left  = succ->right;
            dealloc(succ->key); dealloc(succ->val); dealloc((void*)succ);
            t->len = t->len - 1UL;
            return 1;
        }
        if (parent == (struct BSTNode*)0) t->root = replacement;
        else if (went_left)               parent->left  = replacement;
        else                              parent->right = replacement;
        dealloc(cur->key); dealloc(cur->val); dealloc((void*)cur);
        t->len = t->len - 1UL;
        return 1;
    }
}

// ── In-order traversal (iterative using Stack from stack.h would create a dep; use recursion helper) ──
void bst_inorder_(struct BSTNode* n, void* fn) {
    if (n == (struct BSTNode*)0) return;
    bst_inorder_(n->left, fn);
    unsafe {
        void (*f)(const void*, void*) = (void (*)(const void*, void*))fn;
        f((const void*)n->key, n->val);
    }
    bst_inorder_(n->right, fn);
}

void bst_foreach_inorder(struct BST* t, void* fn) {
    bst_inorder_(t->root, fn);
}

generic<T>
int bst_insert_t(struct BST* t, const void* key, T val) {
    return bst_insert(t, key, (const void*)&val);
}

generic<T>
T* bst_get_t(struct BST* t, const void* key) {
    return (T*)bst_get(t, key);
}
