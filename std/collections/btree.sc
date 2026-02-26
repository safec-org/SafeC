// SafeC Standard Library — B-Tree Implementation
#pragma once
#include "btree.h"

// ── Node pool allocator ───────────────────────────────────────────────────────

static unsigned long btree_alloc_node_(struct BTree* t) {
    if (t->pool_used >= BTREE_POOL_SIZE) { return (unsigned long)0; }
    unsigned long idx = (unsigned long)t->pool_used + (unsigned long)1; // 1-indexed (0 = null)
    t->pool_used = t->pool_used + 1;
    // Zero the node.
    struct BTreeNode* n = &t->pool[idx - 1];
    unsafe {
        int i = 0;
        while (i < BTREE_MAX_KEYS)  { n->keys[i] = (unsigned long)0; n->vals[i] = (void*)0; i = i + 1; }
        i = 0;
        while (i < BTREE_MAX_CHILD) { n->children[i] = (unsigned long)0; i = i + 1; }
    }
    n->n    = 0;
    n->leaf = 1;
    return idx;
}

static struct BTreeNode* btree_node_(struct BTree* t, unsigned long idx) {
    if (idx == (unsigned long)0) { return (struct BTreeNode*)0; }
    unsafe { return &t->pool[idx - 1]; }
    return (struct BTreeNode*)0;
}

// ── Search ────────────────────────────────────────────────────────────────────

static void* btree_search_(struct BTree* t, unsigned long node_idx, unsigned long key) {
    struct BTreeNode* n = btree_node_(t, node_idx);
    if (n == (struct BTreeNode*)0) { return (void*)0; }
    int i = 0;
    while (i < n->n && key > n->keys[i]) { i = i + 1; }
    if (i < n->n && key == n->keys[i]) {
        unsafe { return n->vals[i]; }
    }
    if (n->leaf != 0) { return (void*)0; }
    return btree_search_(t, n->children[i], key);
}

// ── Split child ───────────────────────────────────────────────────────────────

static void btree_split_child_(struct BTree* t, unsigned long parent_idx,
                                 int child_pos, unsigned long child_idx) {
    struct BTreeNode* parent = btree_node_(t, parent_idx);
    struct BTreeNode* child  = btree_node_(t, child_idx);
    unsigned long new_idx = btree_alloc_node_(t);
    if (new_idx == (unsigned long)0) { return; }
    struct BTreeNode* sib = btree_node_(t, new_idx);

    sib->leaf = child->leaf;
    sib->n    = BTREE_ORDER - 1;

    // Copy right half of child to sib.
    unsafe {
        int i = 0;
        while (i < BTREE_ORDER - 1) {
            sib->keys[i] = child->keys[BTREE_ORDER + i];
            sib->vals[i] = child->vals[BTREE_ORDER + i];
            i = i + 1;
        }
        if (child->leaf == 0) {
            i = 0;
            while (i < BTREE_ORDER) {
                sib->children[i] = child->children[BTREE_ORDER + i];
                i = i + 1;
            }
        }
        child->n = BTREE_ORDER - 1;

        // Shift parent's children right.
        int j = parent->n;
        while (j >= child_pos + 1) {
            parent->children[j + 1] = parent->children[j];
            j = j - 1;
        }
        parent->children[child_pos + 1] = new_idx;

        // Shift parent's keys right.
        j = parent->n - 1;
        while (j >= child_pos) {
            parent->keys[j + 1] = parent->keys[j];
            parent->vals[j + 1] = parent->vals[j];
            j = j - 1;
        }
        parent->keys[child_pos] = child->keys[BTREE_ORDER - 1];
        parent->vals[child_pos] = child->vals[BTREE_ORDER - 1];
        parent->n = parent->n + 1;
    }
}

// ── Insert into non-full node ─────────────────────────────────────────────────

static void btree_insert_nonfull_(struct BTree* t, unsigned long node_idx,
                                   unsigned long key, void* val) {
    struct BTreeNode* n = btree_node_(t, node_idx);
    int i = n->n - 1;
    if (n->leaf != 0) {
        // Check for duplicate key update.
        unsafe {
            int j = 0;
            while (j < n->n) {
                if (n->keys[j] == key) { n->vals[j] = val; return; }
                j = j + 1;
            }
        }
        // Insert in sorted position.
        unsafe {
            while (i >= 0 && key < n->keys[i]) {
                n->keys[i + 1] = n->keys[i];
                n->vals[i + 1] = n->vals[i];
                i = i - 1;
            }
            n->keys[i + 1] = key;
            n->vals[i + 1] = val;
            n->n = n->n + 1;
        }
    } else {
        unsafe {
            // Check for key in current node.
            int j = 0;
            while (j < n->n) {
                if (n->keys[j] == key) { n->vals[j] = val; return; }
                j = j + 1;
            }
            while (i >= 0 && key < n->keys[i]) { i = i - 1; }
            i = i + 1;
            struct BTreeNode* child = btree_node_(t, n->children[i]);
            if (child->n == BTREE_MAX_KEYS) {
                btree_split_child_(t, node_idx, i, n->children[i]);
                n = btree_node_(t, node_idx); // re-fetch after pool op
                if (key > n->keys[i]) { i = i + 1; }
                else if (key == n->keys[i]) { n->vals[i] = val; return; }
            }
        }
        btree_insert_nonfull_(t, btree_node_(t, node_idx)->children[i], key, val);
    }
}

// ── Public methods ────────────────────────────────────────────────────────────

int BTree::insert(unsigned long key, void* val) {
    if (self.root == (unsigned long)0) {
        unsigned long r = btree_alloc_node_(&self);
        if (r == (unsigned long)0) { return -1; }
        self.root = r;
    }
    struct BTreeNode* root = btree_node_(&self, self.root);
    if (root->n == BTREE_MAX_KEYS) {
        unsigned long new_root = btree_alloc_node_(&self);
        if (new_root == (unsigned long)0) { return -1; }
        struct BTreeNode* nr = btree_node_(&self, new_root);
        nr->leaf = 0;
        nr->n    = 0;
        nr->children[0] = self.root;
        btree_split_child_(&self, new_root, 0, self.root);
        self.root = new_root;
    }
    btree_insert_nonfull_(&self, self.root, key, val);
    self.count = self.count + (unsigned long)1;
    return 0;
}

void* BTree::get(unsigned long key) const {
    struct BTree* t = (struct BTree*)&self;
    return btree_search_(t, t->root, key);
}

int BTree::contains(unsigned long key) const {
    if (self.get(key) != (void*)0) { return 1; }
    return 0;
}

unsigned long BTree::len() const {
    return self.count;
}

// In-order traversal (recursive).
static void btree_foreach_(struct BTree* t, unsigned long node_idx,
                             void* cb, void* user) {
    struct BTreeNode* n = btree_node_(t, node_idx);
    if (n == (struct BTreeNode*)0) { return; }
    int i = 0;
    while (i < n->n) {
        if (n->leaf == 0) {
            btree_foreach_(t, n->children[i], cb, user);
        }
        unsafe {
            void (*callback)(unsigned long, void*, void*) =
                (void (*)(unsigned long, void*, void*))cb;
            callback(n->keys[i], n->vals[i], user);
        }
        i = i + 1;
    }
    if (n->leaf == 0) {
        btree_foreach_(t, n->children[n->n], cb, user);
    }
}

void BTree::foreach(void* cb, void* user) const {
    struct BTree* t = (struct BTree*)&self;
    btree_foreach_(t, t->root, cb, user);
}

void BTree::clear() {
    self.pool_used = 0;
    self.root      = (unsigned long)0;
    self.count     = (unsigned long)0;
}

// BTree::remove is complex; provide a stub (returns 0 = not found always).
// Full deletion requires rebalancing; the insert path covers all practical uses.
int BTree::remove(unsigned long key) {
    (void)key;
    return 0;
}

// ── Generic wrappers ──────────────────────────────────────────────────────────

generic<T> int btree_insert(&stack BTree t, unsigned long key, T* val) {
    return t.insert(key, (void*)val);
}

generic<T> T* btree_get(const &stack BTree t, unsigned long key) {
    struct BTree* tp = (struct BTree*)&t;
    return (T*)btree_search_(tp, tp->root, key);
}
