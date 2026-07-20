// SafeC Standard Library — B-Tree Implementation
#pragma once
#include <std/collections/btree.h>

// ── Node pool allocator ───────────────────────────────────────────────────────

namespace std {

inline void btree_init(&stack BTree t) {
    t.pool_used = 0;
    t.root      = (unsigned long)0;
    t.count     = (unsigned long)0;
}

static unsigned long btree_alloc_node_(struct BTree* t) {
    unsafe {
        if (t->pool_used >= BTREE_POOL_SIZE) { return (unsigned long)0; }
        unsigned long idx = (unsigned long)t->pool_used + (unsigned long)1; // 1-indexed (0 = null)
        t->pool_used = t->pool_used + 1;
        // Zero the node.
        struct BTreeNode* n = (struct BTreeNode*)&t->pool[idx - (unsigned long)1];
        int i = 0;
        while (i < BTREE_MAX_KEYS)  { n->keys[i] = (unsigned long)0; n->vals[i] = (void*)0; i = i + 1; }
        i = 0;
        while (i < BTREE_MAX_CHILD) { n->children[i] = (unsigned long)0; i = i + 1; }
        n->n    = 0;
        n->leaf = 1;
        return idx;
    }
}

static struct BTreeNode* btree_node_(struct BTree* t, unsigned long idx) {
    unsafe {
        if (idx == (unsigned long)0) { return (struct BTreeNode*)0; }
        return (struct BTreeNode*)&t->pool[idx - (unsigned long)1];
    }
}

// ── Search ────────────────────────────────────────────────────────────────────

static void* btree_search_(struct BTree* t, unsigned long node_idx, unsigned long key) {
    struct BTreeNode* n = btree_node_(t, node_idx);
    unsafe {
        if (n == (struct BTreeNode*)0) { return (void*)0; }
        int i = 0;
        while (i < n->n && key > n->keys[i]) { i = i + 1; }
        if (i < n->n && key == n->keys[i]) {
            return n->vals[i];
        }
        if (n->leaf != 0) { return (void*)0; }
        return btree_search_(t, n->children[i], key);
    }
}

// ── Split child ───────────────────────────────────────────────────────────────

static void btree_split_child_(struct BTree* t, unsigned long parent_idx,
                                 int child_pos, unsigned long child_idx) {
    struct BTreeNode* parent = btree_node_(t, parent_idx);
    struct BTreeNode* child  = btree_node_(t, child_idx);
    unsigned long new_idx = btree_alloc_node_(t);
    if (new_idx == (unsigned long)0) { return; }
    struct BTreeNode* sib = btree_node_(t, new_idx);

    // Copy right half of child to sib.
    unsafe {
        sib->leaf = child->leaf;
        sib->n    = BTREE_ORDER - 1;
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
    unsafe {
        int i = n->n - 1;
        if (n->leaf != 0) {
            // Check for duplicate key update.
            int j = 0;
            while (j < n->n) {
                if (n->keys[j] == key) { n->vals[j] = val; return; }
                j = j + 1;
            }
            // Insert in sorted position.
            while (i >= 0 && key < n->keys[i]) {
                n->keys[i + 1] = n->keys[i];
                n->vals[i + 1] = n->vals[i];
                i = i - 1;
            }
            n->keys[i + 1] = key;
            n->vals[i + 1] = val;
            n->n = n->n + 1;
        } else {
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
            btree_insert_nonfull_(t, btree_node_(t, node_idx)->children[i], key, val);
        }
    }
}

// ── Public methods ────────────────────────────────────────────────────────────

int BTree::insert(unsigned long key, void* val) {
    unsafe {
        struct BTree* t = (struct BTree*)self;
        if (self.root == (unsigned long)0) {
            unsigned long r = btree_alloc_node_(t);
            if (r == (unsigned long)0) { return -1; }
            self.root = r;
        }
        struct BTreeNode* root = btree_node_(t, self.root);
        if (root->n == BTREE_MAX_KEYS) {
            unsigned long new_root = btree_alloc_node_(t);
            if (new_root == (unsigned long)0) { return -1; }
            struct BTreeNode* nr = btree_node_(t, new_root);
            nr->leaf = 0;
            nr->n    = 0;
            nr->children[0] = self.root;
            btree_split_child_(t, new_root, 0, self.root);
            self.root = new_root;
        }
        btree_insert_nonfull_(t, self.root, key, val);
        self.count = self.count + (unsigned long)1;
        return 0;
    }
}

inline void* BTree::get(unsigned long key) const {
    unsafe {
        struct BTree* t = (struct BTree*)self;
        return btree_search_(t, t->root, key);
    }
}

inline int BTree::contains(unsigned long key) const {
    if (self.get(key) != (void*)0) { return 1; }
    return 0;
}

inline unsigned long BTree::len() const {
    return self.count;
}

// In-order traversal (recursive).
static void btree_foreach_(struct BTree* t, unsigned long node_idx,
                             void* cb, void* user) {
    struct BTreeNode* n = btree_node_(t, node_idx);
    unsafe {
        if (n == (struct BTreeNode*)0) { return; }
        int i = 0;
        while (i < n->n) {
            if (n->leaf == 0) {
                btree_foreach_(t, n->children[i], cb, user);
            }
            fn void(unsigned long, void*, void*) callback =
                (fn void(unsigned long, void*, void*))cb;
            callback(n->keys[i], n->vals[i], user);
            i = i + 1;
        }
        if (n->leaf == 0) {
            btree_foreach_(t, n->children[n->n], cb, user);
        }
    }
}

inline void BTree::foreach(void* cb, void* user) const {
    unsafe {
        struct BTree* t = (struct BTree*)self;
        btree_foreach_(t, t->root, cb, user);
    }
}

inline void BTree::clear() {
    self.pool_used = 0;
    self.root      = (unsigned long)0;
    self.count     = (unsigned long)0;
}

// ── Delete ────────────────────────────────────────────────────────────────────
// Standard B-tree deletion (CLRS/geeksforgeeks formulation): a key found in
// a leaf is removed directly; a key found in an internal node is replaced by
// its in-order predecessor or successor (pulled up from whichever adjacent
// child has a spare key to give up without violating the minimum-keys
// invariant) and that replacement key is then recursively deleted from the
// child it came from; and before descending into any child during the
// search for a key that isn't in the current node, that child is first
// topped up to at least BTREE_ORDER keys (via a borrow from a sibling with
// spare keys, or a merge with a sibling that has none) so the recursion
// never has to delete out of a node already at the minimum.
//
// Deleted nodes are not returned to a free list — the pool (BTree.pool,
// sized BTREE_POOL_SIZE) only ever grows via btree_alloc_node_, so a tree
// under heavy insert/delete churn can still exhaust the pool even though
// its live key count stays low. Reclaiming freed slots would need a
// free-list field added to 'struct BTree' (a public API/layout change);
// out of scope for fixing the correctness bug this replaces (remove()
// silently doing nothing and still reporting "removed").

static int btree_find_idx_(struct BTreeNode* n, unsigned long key) {
    unsafe {
        int idx = 0;
        while (idx < n->n && n->keys[idx] < key) { idx = idx + 1; }
        return idx;
    }
}

// Max key (and its value) in the subtree rooted at node_idx — always the
// last key of the rightmost leaf.
static void btree_get_pred_(struct BTree* t, unsigned long node_idx,
                             &stack unsigned long outKey, &stack void* outVal) {
    unsafe {
        struct BTreeNode* cur = btree_node_(t, node_idx);
        while (cur->leaf == 0) {
            cur = btree_node_(t, cur->children[cur->n]);
        }
        *outKey = cur->keys[cur->n - 1];
        *outVal = cur->vals[cur->n - 1];
    }
}

// Min key (and its value) in the subtree rooted at node_idx — always the
// first key of the leftmost leaf.
static void btree_get_succ_(struct BTree* t, unsigned long node_idx,
                             &stack unsigned long outKey, &stack void* outVal) {
    unsafe {
        struct BTreeNode* cur = btree_node_(t, node_idx);
        while (cur->leaf == 0) {
            cur = btree_node_(t, cur->children[0]);
        }
        *outKey = cur->keys[0];
        *outVal = cur->vals[0];
    }
}

// Merge node->children[idx], node->keys[idx]/vals[idx] (the separator), and
// node->children[idx+1] into a single node at node->children[idx]. Both
// children must have exactly BTREE_ORDER-1 keys (the merge-eligible case),
// so the combined key count is exactly BTREE_MAX_KEYS — fits without
// overflow. node loses one key and one child pointer.
static void btree_merge_(struct BTree* t, unsigned long node_idx, int idx) {
    unsafe {
        struct BTreeNode* node  = btree_node_(t, node_idx);
        struct BTreeNode* left  = btree_node_(t, node->children[idx]);
        struct BTreeNode* right = btree_node_(t, node->children[idx + 1]);

        left->keys[left->n] = node->keys[idx];
        left->vals[left->n] = node->vals[idx];

        int i = 0;
        while (i < right->n) {
            left->keys[left->n + 1 + i] = right->keys[i];
            left->vals[left->n + 1 + i] = right->vals[i];
            i = i + 1;
        }
        if (left->leaf == 0) {
            i = 0;
            while (i <= right->n) {
                left->children[left->n + 1 + i] = right->children[i];
                i = i + 1;
            }
        }
        left->n = left->n + 1 + right->n;

        i = idx;
        while (i < node->n - 1) {
            node->keys[i] = node->keys[i + 1];
            node->vals[i] = node->vals[i + 1];
            i = i + 1;
        }
        i = idx + 1;
        while (i < node->n) {
            node->children[i] = node->children[i + 1];
            i = i + 1;
        }
        node->n = node->n - 1;
    }
}

// Move node->children[idx-1]'s last key up through node into the front of
// node->children[idx] (used when the left sibling has a spare key).
static void btree_borrow_from_prev_(struct BTree* t, unsigned long node_idx, int idx) {
    unsafe {
        struct BTreeNode* node  = btree_node_(t, node_idx);
        struct BTreeNode* child = btree_node_(t, node->children[idx]);
        struct BTreeNode* sib   = btree_node_(t, node->children[idx - 1]);

        int i = child->n - 1;
        while (i >= 0) {
            child->keys[i + 1] = child->keys[i];
            child->vals[i + 1] = child->vals[i];
            i = i - 1;
        }
        if (child->leaf == 0) {
            i = child->n;
            while (i >= 0) {
                child->children[i + 1] = child->children[i];
                i = i - 1;
            }
        }
        child->keys[0] = node->keys[idx - 1];
        child->vals[0] = node->vals[idx - 1];
        if (child->leaf == 0) {
            child->children[0] = sib->children[sib->n];
        }
        node->keys[idx - 1] = sib->keys[sib->n - 1];
        node->vals[idx - 1] = sib->vals[sib->n - 1];

        child->n = child->n + 1;
        sib->n   = sib->n - 1;
    }
}

// Move node->children[idx+1]'s first key up through node into the back of
// node->children[idx] (used when the right sibling has a spare key).
static void btree_borrow_from_next_(struct BTree* t, unsigned long node_idx, int idx) {
    unsafe {
        struct BTreeNode* node  = btree_node_(t, node_idx);
        struct BTreeNode* child = btree_node_(t, node->children[idx]);
        struct BTreeNode* sib   = btree_node_(t, node->children[idx + 1]);

        child->keys[child->n] = node->keys[idx];
        child->vals[child->n] = node->vals[idx];
        if (child->leaf == 0) {
            child->children[child->n + 1] = sib->children[0];
        }

        node->keys[idx] = sib->keys[0];
        node->vals[idx] = sib->vals[0];

        int i = 0;
        while (i < sib->n - 1) {
            sib->keys[i] = sib->keys[i + 1];
            sib->vals[i] = sib->vals[i + 1];
            i = i + 1;
        }
        if (sib->leaf == 0) {
            i = 0;
            while (i < sib->n) {
                sib->children[i] = sib->children[i + 1];
                i = i + 1;
            }
        }

        child->n = child->n + 1;
        sib->n   = sib->n - 1;
    }
}

// Ensure node->children[idx] has at least BTREE_ORDER keys before the
// caller descends into it, by borrowing from whichever neighbor has spare
// keys, or merging with a neighbor if neither does.
static void btree_fill_(struct BTree* t, unsigned long node_idx, int idx) {
    unsafe {
        struct BTreeNode* node = btree_node_(t, node_idx);
        if (idx != 0) {
            struct BTreeNode* prevSib = btree_node_(t, node->children[idx - 1]);
            if (prevSib->n >= BTREE_ORDER) {
                btree_borrow_from_prev_(t, node_idx, idx);
                return;
            }
        }
        if (idx != node->n) {
            struct BTreeNode* nextSib = btree_node_(t, node->children[idx + 1]);
            if (nextSib->n >= BTREE_ORDER) {
                btree_borrow_from_next_(t, node_idx, idx);
                return;
            }
        }
        if (idx != node->n) {
            btree_merge_(t, node_idx, idx);
        } else {
            btree_merge_(t, node_idx, idx - 1);
        }
    }
}

static int btree_remove_(struct BTree* t, unsigned long node_idx, unsigned long key) {
    unsafe {
        struct BTreeNode* node = btree_node_(t, node_idx);
        int idx = btree_find_idx_(node, key);

        if (idx < node->n && node->keys[idx] == key) {
            if (node->leaf != 0) {
                int i = idx;
                while (i < node->n - 1) {
                    node->keys[i] = node->keys[i + 1];
                    node->vals[i] = node->vals[i + 1];
                    i = i + 1;
                }
                node->n = node->n - 1;
                return 1;
            }

            struct BTreeNode* leftChild  = btree_node_(t, node->children[idx]);
            struct BTreeNode* rightChild = btree_node_(t, node->children[idx + 1]);
            if (leftChild->n >= BTREE_ORDER) {
                unsigned long predKey = (unsigned long)0;
                void* predVal = (void*)0;
                btree_get_pred_(t, node->children[idx], &predKey, &predVal);
                struct BTreeNode* n2 = btree_node_(t, node_idx);
                n2->keys[idx] = predKey;
                n2->vals[idx] = predVal;
                btree_remove_(t, n2->children[idx], predKey);
            } else if (rightChild->n >= BTREE_ORDER) {
                unsigned long succKey = (unsigned long)0;
                void* succVal = (void*)0;
                btree_get_succ_(t, node->children[idx + 1], &succKey, &succVal);
                struct BTreeNode* n2 = btree_node_(t, node_idx);
                n2->keys[idx] = succKey;
                n2->vals[idx] = succVal;
                btree_remove_(t, n2->children[idx + 1], succKey);
            } else {
                btree_merge_(t, node_idx, idx);
                struct BTreeNode* n2 = btree_node_(t, node_idx);
                btree_remove_(t, n2->children[idx], key);
            }
            return 1;
        }

        if (node->leaf != 0) {
            return 0; // key not present anywhere in the tree
        }

        int flag = 0;
        if (idx == node->n) { flag = 1; }
        struct BTreeNode* child = btree_node_(t, node->children[idx]);
        if (child->n < BTREE_ORDER) {
            btree_fill_(t, node_idx, idx);
        }
        struct BTreeNode* n2 = btree_node_(t, node_idx);
        if (flag != 0 && idx > n2->n) {
            return btree_remove_(t, n2->children[idx - 1], key);
        }
        return btree_remove_(t, n2->children[idx], key);
    }
}

int BTree::remove(unsigned long key) {
    unsafe {
        struct BTree* t = (struct BTree*)self;
        if (self.root == (unsigned long)0) { return 0; }

        int found = btree_remove_(t, self.root, key);
        if (found == 0) { return 0; }

        struct BTreeNode* root = btree_node_(t, self.root);
        if (root->n == 0) {
            if (root->leaf == 0) {
                self.root = root->children[0];
            } else {
                self.root = (unsigned long)0;
            }
        }
        self.count = self.count - (unsigned long)1;
        return 1;
    }
}

// ── Generic wrappers ──────────────────────────────────────────────────────────

generic<T> int btree_insert(&stack BTree t, unsigned long key, T* val) {
    return t.insert(key, (void*)val);
}

generic<T> T* btree_get(const &stack BTree t, unsigned long key) {
    unsafe {
        struct BTree* tp = (struct BTree*)t;
        return (T*)btree_search_(tp, tp->root, key);
    }
}

} // namespace std
