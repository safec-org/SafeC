#pragma once
// SafeC Standard Library — BST<K,V> (unbalanced binary search tree)
//
//   struct BST<int, int, Less<int>> t = std::BST::new();   // natural order
//   struct BST<Name, int, FnOrder<Name>> u = std::BST::new_with(std::FnOrder::of(cmp_name));
//   t.insert(50, 500);
//   match (t.get(50)) { case some(v): *v = 501; case null: ... }
//   t.foreach_inorder(print_pair);   // ascending key order
//   t.free();
//
// C is a compile-time ordering, an Order (see std/cmp.h): every key comparison
// is a direct, inlinable call to C::cmp — no function pointer per compare.
// Keys and values are stored by value, one node per entry, drawn from the
// tree's own NodePool (see node_pool.h). Every operation is iterative — no recursion — so a
// degenerate (sorted-insert) tree costs O(n) time per lookup but never
// overflows a small bare-metal stack. get()/min_key()/max_key() hand out
// references into a node, valid until that entry is removed.
#include <std/mem.h>
#include <std/collections/node_pool.h>
#include <std/cmp.h>

namespace std {

generic<K, V>
struct BSTNode {
    K                                 key;
    V                                 val;
    ?&heap struct BSTNode<K, V>       left;
    ?&heap struct BSTNode<K, V>       right;
};

generic<K, V, C: Order>
struct BST {
    ?&heap struct BSTNode<K, V>       root;
    C                                 ord;
    unsigned long                     len;
    struct NodePool<struct BSTNode<K, V>> pool;

    static struct BST<K, V, C> new();            // stateless C (Less, Greater): C from the target
    static struct BST<K, V, C> new_with(C ord);  // ordering value (FnOrder, ...)

    int           insert(K key, V val);    // insert or overwrite; 0 on OOM
    ?&V           get(K key) const;        // null if missing
    int           contains(K key) const;
    ?V            remove(K key);           // the removed value, or null
    unsigned long length() const;
    int           is_empty() const;
    ?&K           min_key() const;         // null if empty
    ?&K           max_key() const;         // null if empty
    void          foreach_inorder(fn void(const &K, &V) func);
    void          clear();
    void          free();

    ?&heap struct BSTNode<K, V> find_(const &K key) const;
    void          relink_(?&heap struct BSTNode<K, V> parent, int dir,
                          ?&heap struct BSTNode<K, V> child);
};

generic<K, V, C: Order>
struct BST<K, V, C> BST::new_with(C ord) {
    struct BST<K, V, C> t;
    t.root = null;
    t.ord = ord;
    t.len = 0UL;
    t.pool = NodePool::new();
    return t;
}

// Stateless orderings (Less, Greater) need no value: C comes from the target.
generic<K, V, C: Order>
struct BST<K, V, C> BST::new() {
    struct BST<K, V, C> t;
    t.root = null;
    t.len = 0UL;
    t.pool = NodePool::new();
    return t;
}

generic<K, V, C: Order>
unsigned long BST::length() const { return self.len; }

generic<K, V, C: Order>
int BST::is_empty() const { return self.len == 0UL; }

generic<K, V, C: Order>
?&heap struct BSTNode<K, V> BST::find_(const &K key) const {
    ?&heap struct BSTNode<K, V> cur = self.root;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                int c = self.ord.cmp(key, &n.key);
                if (c == 0) return cur;
                cur = c < 0 ? n.left : n.right;
            }
            case null: { }
        }
    }
    return null;
}

// Points parent's 'dir' link (-1 left, 1 right; 0 = the root) at child.
generic<K, V, C: Order>
void BST::relink_(?&heap struct BSTNode<K, V> parent, int dir,
                  ?&heap struct BSTNode<K, V> child) {
    match (parent) {
        case null: self.root = child;
        case some(p): {
            if (dir < 0) p.left = child;
            else p.right = child;
        }
    }
}

generic<K, V, C: Order>
int BST::insert(K key, V val) {
    ?&heap struct BSTNode<K, V> parent = null;
    ?&heap struct BSTNode<K, V> cur = self.root;
    int dir = 0;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                int c = self.ord.cmp(&key, &n.key);
                if (c == 0) { n.val = val; return 1; }
                parent = cur;
                dir = c < 0 ? -1 : 1;
                cur = c < 0 ? n.left : n.right;
            }
            case null: { }
        }
    }
    ?&heap struct BSTNode<K, V> nq = self.pool.take();
    match (nq) {
        case null: return 0;
        case some(n): { n.key = key; n.val = val; n.left = null; n.right = null; }
    }
    self.relink_(parent, dir, nq);
    self.len = self.len + 1UL;
    return 1;
}

generic<K, V, C: Order>
?&V BST::get(K key) const {
    match (self.find_(&key)) {
        case some(n): unsafe { return (?&V)&n.val; }
        case null: { }
    }
    return null;
}

generic<K, V, C: Order>
int BST::contains(K key) const { return !self.find_(&key).is_null(); }

generic<K, V, C: Order>
?V BST::remove(K key) {
    ?&heap struct BSTNode<K, V> parent = null;
    ?&heap struct BSTNode<K, V> cur = self.root;
    int dir = 0;
    int found = 0;
    while (!found && !cur.is_null()) {
        match (cur) {
            case some(n): {
                int c = self.ord.cmp(&key, &n.key);
                if (c == 0) { found = 1; }
                else {
                    parent = cur;
                    dir = c < 0 ? -1 : 1;
                    cur = c < 0 ? n.left : n.right;
                }
            }
            case null: { }
        }
    }
    match (cur) {
        case null: return null;
        case some(n): {
            V out = n.val;
            if (!n.left.is_null() && !n.right.is_null()) {
                // Two children: pull up the in-order successor (leftmost
                // node of the right subtree, which has no left child).
                ?&heap struct BSTNode<K, V> sp = cur;
                int sdir = 1;
                ?&heap struct BSTNode<K, V> s = n.right;
                int more = 1;
                while (more) {
                    match (s) {
                        case some(sn): {
                            if (sn.left.is_null()) more = 0;
                            else { sp = s; sdir = -1; s = sn.left; }
                        }
                        case null: more = 0;
                    }
                }
                match (s) {
                    case some(sn): {
                        n.key = sn.key;
                        n.val = sn.val;
                        self.relink_(sp, sdir, sn.right);
                        self.pool.give(sn);
                    }
                    case null: { }
                }
            } else {
                self.relink_(parent, dir, n.left.is_null() ? n.right : n.left);
                self.pool.give(n);
            }
            self.len = self.len - 1UL;
            return out;
        }
    }
    return null;
}

generic<K, V, C: Order>
?&K BST::min_key() const {
    ?&heap struct BSTNode<K, V> cur = self.root;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                if (n.left.is_null()) unsafe { return (?&K)&n.key; }
                cur = n.left;
            }
            case null: { }
        }
    }
    return null;
}

generic<K, V, C: Order>
?&K BST::max_key() const {
    ?&heap struct BSTNode<K, V> cur = self.root;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                if (n.right.is_null()) unsafe { return (?&K)&n.key; }
                cur = n.right;
            }
            case null: { }
        }
    }
    return null;
}

// Morris in-order traversal: O(n) time, O(1) extra space. Threads each
// predecessor's empty right link back to its successor while descending and
// removes the thread on the way back, so the tree is unchanged afterwards.
generic<K, V, C: Order>
void BST::foreach_inorder(fn void(const &K, &V) func) {
    ?&heap struct BSTNode<K, V> cur = self.root;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                match (n.left) {
                    case null: {
                        unsafe { func(&n.key, &n.val); }
                        cur = n.right;
                    }
                    case some(l): {
                        ?&heap struct BSTNode<K, V> pre = l;
                        int walking = 1;
                        while (walking) {
                            match (pre) {
                                case some(p): {
                                    match (p.right) {
                                        case null: walking = 0;
                                        case some(r): {
                                            int back = 0;
                                            unsafe { back = (void*)r == (void*)n; }
                                            if (back) walking = 0;
                                            else pre = r;
                                        }
                                    }
                                }
                                case null: walking = 0;
                            }
                        }
                        match (pre) {
                            case some(p): {
                                if (p.right.is_null()) {
                                    p.right = cur;            // thread back to n
                                    cur = n.left;
                                } else {
                                    p.right = null;           // remove the thread
                                    unsafe { func(&n.key, &n.val); }
                                    cur = n.right;
                                }
                            }
                            case null: { }
                        }
                    }
                }
            }
            case null: { }
        }
    }
}

generic<K, V, C: Order>
void BST::free() {
    self.pool.free_all();
    self.root = null;
    self.len = 0UL;
}

generic<K, V, C: Order>
void BST::clear() { self.free(); }

} // namespace std
