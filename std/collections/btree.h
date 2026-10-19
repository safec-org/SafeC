// SafeC Standard Library — BTree<V>: B-tree ordered map, unsigned long keys
//
//   ?&BTree<int> tq = std::alloc(sizeof(struct BTree<int>));  // ~pool-sized; keep it off the stack
//   &BTree<int> t = std::expect_ref(tq, "oom");
//   std::btree_init(t);
//   t.insert(42UL, 7);
//   int v = t.get(42UL).default(0);
//
// Deterministic O(log n) lookups with no heap use after construction: nodes
// come from a fixed in-struct pool of BTREE_POOL_SIZE nodes (minimum degree
// BTREE_ORDER, so up to BTREE_MAX_KEYS keys per node), and nodes freed by a
// merge go on a free list for reuse. Values are stored by value; get() hands
// out a '?&V' into the pool, valid until the next insert()/remove().
#pragma once

#define BTREE_ORDER      4      // min keys per non-root node
#define BTREE_MAX_KEYS   7      // 2*ORDER - 1
#define BTREE_MAX_CHILD  8      // 2*ORDER
#define BTREE_POOL_SIZE  256    // max nodes in the pool

namespace std {

generic<V>
struct BTreeNode {
    unsigned long keys[BTREE_MAX_KEYS];
    V             vals[BTREE_MAX_KEYS];
    unsigned long children[BTREE_MAX_CHILD]; // pool index + 1 (0 = none)
    int           n;            // number of keys
    int           leaf;         // 1 if leaf node
};

generic<V>
struct BTree {
    struct BTreeNode<V> pool[BTREE_POOL_SIZE];
    int                 pool_used;  // high-water mark
    unsigned long       free_head;  // freed nodes, chained through children[0]
    unsigned long       root;       // pool index + 1 (0 = empty tree)
    unsigned long       count;      // key-value pairs

    int           insert(unsigned long key, V val);  // 0 on success, -1 if the pool is full
    ?&V           get(unsigned long key) const;      // null if missing
    int           contains(unsigned long key) const;
    ?V            remove(unsigned long key);         // the removed value, or null
    unsigned long len() const;
    void          foreach(fn void(unsigned long, &V) cb);  // ascending key order
    void          clear();

    // ── Internal ────────────────────────────────────────────────────────────
    &BTreeNode<V> node_(unsigned long idx) const;
    unsigned long alloc_node_();
    void          free_node_(unsigned long idx);
    int           split_child_(unsigned long parent_idx, int pos);
    void          merge_(unsigned long node_idx, int idx);
    void          borrow_prev_(unsigned long node_idx, int idx);
    void          borrow_next_(unsigned long node_idx, int idx);
    void          fill_(unsigned long node_idx, int idx);
    int           remove_(unsigned long node_idx, unsigned long key);
    void          foreach_(unsigned long node_idx, fn void(unsigned long, &V) cb);
};

// Initialise (or reset) an empty tree in place. Only the scalar fields are
// touched — pool nodes are claimed lazily, so a ~pool-sized memset (or a
// by-value constructor copying the whole pool) would be pure waste.
generic<V>
void btree_init(&BTree<V> t) {
    t.pool_used = 0;
    t.free_head = 0UL;
    t.root      = 0UL;
    t.count     = 0UL;
}

generic<V>
&BTreeNode<V> BTree::node_(unsigned long idx) const {
    unsafe { return &self.pool[idx - 1UL]; }
}

generic<V>
unsigned long BTree::alloc_node_() {
    unsigned long idx = 0UL;
    if (self.free_head != 0UL) {
        idx = self.free_head;
        unsafe { self.free_head = self.pool[idx - 1UL].children[0]; }
    } else {
        if (self.pool_used >= BTREE_POOL_SIZE) return 0UL;
        self.pool_used = self.pool_used + 1;
        idx = (unsigned long)self.pool_used;
    }
    &BTreeNode<V> n = self.node_(idx);
    unsafe {
        int i = 0;
        while (i < BTREE_MAX_CHILD) { n.children[i] = 0UL; i = i + 1; }
    }
    n.n = 0;
    n.leaf = 1;
    return idx;
}

generic<V>
void BTree::free_node_(unsigned long idx) {
    &BTreeNode<V> n = self.node_(idx);
    unsafe { n.children[0] = self.free_head; }
    self.free_head = idx;
}

generic<V>
unsigned long BTree::len() const { return self.count; }

// ── Search ────────────────────────────────────────────────────────────────────

generic<V>
?&V BTree::get(unsigned long key) const {
    unsigned long cur = self.root;
    while (cur != 0UL) {
        &BTreeNode<V> n = self.node_(cur);
        unsafe {
            int i = 0;
            while (i < n.n && key > n.keys[i]) { i = i + 1; }
            if (i < n.n && key == n.keys[i]) return (?&V)&n.vals[i];
            if (n.leaf != 0) return null;
            cur = n.children[i];
        }
    }
    return null;
}

generic<V>
int BTree::contains(unsigned long key) const { return !self.get(key).is_null(); }

// ── Insert ────────────────────────────────────────────────────────────────────

// Splits the full child at parent.children[pos] around its median key,
// which moves up into parent. 0 if the pool is exhausted.
generic<V>
int BTree::split_child_(unsigned long parent_idx, int pos) {
    unsigned long sib_idx = self.alloc_node_();
    if (sib_idx == 0UL) return 0;
    &BTreeNode<V> parent = self.node_(parent_idx);
    unsigned long child_idx = 0UL;
    unsafe { child_idx = parent.children[pos]; }
    &BTreeNode<V> child = self.node_(child_idx);
    &BTreeNode<V> sib = self.node_(sib_idx);
    unsafe {
        sib.leaf = child.leaf;
        sib.n    = BTREE_ORDER - 1;
        int i = 0;
        while (i < BTREE_ORDER - 1) {
            sib.keys[i] = child.keys[BTREE_ORDER + i];
            sib.vals[i] = child.vals[BTREE_ORDER + i];
            i = i + 1;
        }
        if (child.leaf == 0) {
            i = 0;
            while (i < BTREE_ORDER) { sib.children[i] = child.children[BTREE_ORDER + i]; i = i + 1; }
        }
        child.n = BTREE_ORDER - 1;

        int j = parent.n;
        while (j >= pos + 1) { parent.children[j + 1] = parent.children[j]; j = j - 1; }
        parent.children[pos + 1] = sib_idx;
        j = parent.n - 1;
        while (j >= pos) {
            parent.keys[j + 1] = parent.keys[j];
            parent.vals[j + 1] = parent.vals[j];
            j = j - 1;
        }
        parent.keys[pos] = child.keys[BTREE_ORDER - 1];
        parent.vals[pos] = child.vals[BTREE_ORDER - 1];
        parent.n = parent.n + 1;
    }
    return 1;
}

generic<V>
int BTree::insert(unsigned long key, V val) {
    // Overwrite in place first: no splits for a key that's already present.
    match (self.get(key)) {
        case some(slot): { *slot = val; return 0; }
        case null: { }
    }
    if (self.root == 0UL) {
        unsigned long r = self.alloc_node_();
        if (r == 0UL) return -1;
        self.root = r;
    }
    if (self.node_(self.root).n == BTREE_MAX_KEYS) {
        unsigned long nr = self.alloc_node_();
        if (nr == 0UL) return -1;
        &BTreeNode<V> top = self.node_(nr);
        top.leaf = 0;
        unsafe { top.children[0] = self.root; }
        if (!self.split_child_(nr, 0)) { self.free_node_(nr); return -1; }
        self.root = nr;
    }
    // Descend, splitting full children before entering them.
    unsigned long cur = self.root;
    while (1) {
        &BTreeNode<V> n = self.node_(cur);
        int i = n.n - 1;
        if (n.leaf != 0) {
            unsafe {
                while (i >= 0 && key < n.keys[i]) {
                    n.keys[i + 1] = n.keys[i];
                    n.vals[i + 1] = n.vals[i];
                    i = i - 1;
                }
                n.keys[i + 1] = key;
                n.vals[i + 1] = val;
            }
            n.n = n.n + 1;
            self.count = self.count + 1UL;
            return 0;
        }
        unsafe { while (i >= 0 && key < n.keys[i]) { i = i - 1; } }
        i = i + 1;
        unsigned long child = 0UL;
        unsafe { child = n.children[i]; }
        if (self.node_(child).n == BTREE_MAX_KEYS) {
            if (!self.split_child_(cur, i)) return -1;
            unsafe {
                if (key > n.keys[i]) i = i + 1;
                child = n.children[i];
            }
        }
        cur = child;
    }
    return -1;
}

// ── Traversal ─────────────────────────────────────────────────────────────────

generic<V>
void BTree::foreach_(unsigned long node_idx, fn void(unsigned long, &V) cb) {
    if (node_idx == 0UL) return;
    &BTreeNode<V> n = self.node_(node_idx);
    unsafe {
        int i = 0;
        while (i < n.n) {
            if (n.leaf == 0) self.foreach_(n.children[i], cb);
            cb(n.keys[i], &n.vals[i]);
            i = i + 1;
        }
        if (n.leaf == 0) self.foreach_(n.children[n.n], cb);
    }
}

generic<V>
void BTree::foreach(fn void(unsigned long, &V) cb) { self.foreach_(self.root, cb); }

generic<V>
void BTree::clear() {
    self.pool_used = 0;
    self.free_head = 0UL;
    self.root      = 0UL;
    self.count     = 0UL;
}

// ── Delete ────────────────────────────────────────────────────────────────────
// Standard B-tree deletion (CLRS): a key in a leaf is removed directly; a key
// in an internal node is replaced by its in-order predecessor or successor
// (from whichever adjacent child can spare a key) and that key is deleted
// from the child; before descending into a child for a key not in the
// current node, the child is topped up to at least BTREE_ORDER keys (borrow
// from a sibling, or merge with one) so deletion never underflows a node.

// Merges children[idx], the separator keys[idx] and children[idx+1] into
// children[idx] (both children hold BTREE_ORDER-1 keys, so the result is
// exactly BTREE_MAX_KEYS); the emptied right node goes back on the free list.
generic<V>
void BTree::merge_(unsigned long node_idx, int idx) {
    &BTreeNode<V> node = self.node_(node_idx);
    unsigned long right_idx = 0UL;
    unsafe { right_idx = node.children[idx + 1]; }
    unsafe {
        &BTreeNode<V> left  = self.node_(node.children[idx]);
        &BTreeNode<V> right = self.node_(right_idx);
        left.keys[left.n] = node.keys[idx];
        left.vals[left.n] = node.vals[idx];
        int i = 0;
        while (i < right.n) {
            left.keys[left.n + 1 + i] = right.keys[i];
            left.vals[left.n + 1 + i] = right.vals[i];
            i = i + 1;
        }
        if (left.leaf == 0) {
            i = 0;
            while (i <= right.n) { left.children[left.n + 1 + i] = right.children[i]; i = i + 1; }
        }
        left.n = left.n + 1 + right.n;

        i = idx;
        while (i < node.n - 1) {
            node.keys[i] = node.keys[i + 1];
            node.vals[i] = node.vals[i + 1];
            i = i + 1;
        }
        i = idx + 1;
        while (i < node.n) { node.children[i] = node.children[i + 1]; i = i + 1; }
        node.n = node.n - 1;
    }
    self.free_node_(right_idx);
}

generic<V>
void BTree::borrow_prev_(unsigned long node_idx, int idx) {
    &BTreeNode<V> node = self.node_(node_idx);
    unsafe {
        &BTreeNode<V> child = self.node_(node.children[idx]);
        &BTreeNode<V> sib   = self.node_(node.children[idx - 1]);
        int i = child.n - 1;
        while (i >= 0) {
            child.keys[i + 1] = child.keys[i];
            child.vals[i + 1] = child.vals[i];
            i = i - 1;
        }
        if (child.leaf == 0) {
            i = child.n;
            while (i >= 0) { child.children[i + 1] = child.children[i]; i = i - 1; }
            child.children[0] = sib.children[sib.n];
        }
        child.keys[0] = node.keys[idx - 1];
        child.vals[0] = node.vals[idx - 1];
        node.keys[idx - 1] = sib.keys[sib.n - 1];
        node.vals[idx - 1] = sib.vals[sib.n - 1];
        child.n = child.n + 1;
        sib.n   = sib.n - 1;
    }
}

generic<V>
void BTree::borrow_next_(unsigned long node_idx, int idx) {
    &BTreeNode<V> node = self.node_(node_idx);
    unsafe {
        &BTreeNode<V> child = self.node_(node.children[idx]);
        &BTreeNode<V> sib   = self.node_(node.children[idx + 1]);
        child.keys[child.n] = node.keys[idx];
        child.vals[child.n] = node.vals[idx];
        if (child.leaf == 0) child.children[child.n + 1] = sib.children[0];
        node.keys[idx] = sib.keys[0];
        node.vals[idx] = sib.vals[0];
        int i = 0;
        while (i < sib.n - 1) {
            sib.keys[i] = sib.keys[i + 1];
            sib.vals[i] = sib.vals[i + 1];
            i = i + 1;
        }
        if (sib.leaf == 0) {
            i = 0;
            while (i < sib.n) { sib.children[i] = sib.children[i + 1]; i = i + 1; }
        }
        child.n = child.n + 1;
        sib.n   = sib.n - 1;
    }
}

generic<V>
void BTree::fill_(unsigned long node_idx, int idx) {
    &BTreeNode<V> node = self.node_(node_idx);
    unsafe {
        if (idx != 0 && self.node_(node.children[idx - 1]).n >= BTREE_ORDER) {
            self.borrow_prev_(node_idx, idx);
            return;
        }
        if (idx != node.n && self.node_(node.children[idx + 1]).n >= BTREE_ORDER) {
            self.borrow_next_(node_idx, idx);
            return;
        }
    }
    if (idx != node.n) self.merge_(node_idx, idx);
    else self.merge_(node_idx, idx - 1);
}

generic<V>
int BTree::remove_(unsigned long node_idx, unsigned long key) {
    &BTreeNode<V> node = self.node_(node_idx);
    int idx = 0;
    unsafe { while (idx < node.n && node.keys[idx] < key) { idx = idx + 1; } }
    int here = 0;
    unsafe { here = idx < node.n && node.keys[idx] == key; }

    if (here) {
        if (node.leaf != 0) {
            unsafe {
                int i = idx;
                while (i < node.n - 1) {
                    node.keys[i] = node.keys[i + 1];
                    node.vals[i] = node.vals[i + 1];
                    i = i + 1;
                }
            }
            node.n = node.n - 1;
            return 1;
        }
        unsigned long lc = 0UL;
        unsigned long rc = 0UL;
        unsafe { lc = node.children[idx]; rc = node.children[idx + 1]; }
        if (self.node_(lc).n >= BTREE_ORDER) {
            // predecessor: last key of the rightmost leaf under lc
            unsigned long c = lc;
            while (self.node_(c).leaf == 0) { &BTreeNode<V> cn = self.node_(c); unsafe { c = cn.children[cn.n]; } }
            &BTreeNode<V> leaf = self.node_(c);
            unsafe {
                node.keys[idx] = leaf.keys[leaf.n - 1];
                node.vals[idx] = leaf.vals[leaf.n - 1];
                return self.remove_(lc, node.keys[idx]);
            }
        }
        if (self.node_(rc).n >= BTREE_ORDER) {
            // successor: first key of the leftmost leaf under rc
            unsigned long c = rc;
            while (self.node_(c).leaf == 0) { unsafe { c = self.node_(c).children[0]; } }
            &BTreeNode<V> leaf = self.node_(c);
            unsafe {
                node.keys[idx] = leaf.keys[0];
                node.vals[idx] = leaf.vals[0];
                return self.remove_(rc, node.keys[idx]);
            }
        }
        self.merge_(node_idx, idx);
        return self.remove_(lc, key);
    }

    if (node.leaf != 0) return 0;   // not in the tree

    int last = idx == node.n;
    unsigned long child = 0UL;
    unsafe { child = node.children[idx]; }
    if (self.node_(child).n < BTREE_ORDER) self.fill_(node_idx, idx);
    unsafe {
        if (last && idx > node.n) return self.remove_(node.children[idx - 1], key);
        return self.remove_(node.children[idx], key);
    }
}

generic<V>
?V BTree::remove(unsigned long key) {
    ?V out = null;
    match (self.get(key)) {
        case null: return null;
        case some(v): out = *v;
    }
    self.remove_(self.root, key);
    &BTreeNode<V> r = self.node_(self.root);
    if (r.n == 0) {
        unsigned long old = self.root;
        unsafe { self.root = r.leaf == 0 ? r.children[0] : 0UL; }
        self.free_node_(old);
    }
    self.count = self.count - 1UL;
    return out;
}

} // namespace std
