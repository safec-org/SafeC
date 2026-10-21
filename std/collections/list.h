#pragma once
// SafeC Standard Library — List<T> (doubly linked list)
//
//   struct List<int> l = std::List::new();
//   l.push_back(1); l.push_front(0);
//   ?int first = l.pop_front();     // 0
//   l.free();
//
// One node per element, holding the element by value; nodes come from the
// list's own NodePool (see node_pool.h), so push/pop don't hit the
// allocator per element and free() releases whole chunks. front()/back()/
// find() hand out '?&T' into a node, valid until that node is removed.
#include <std/mem.h>
#include <std/collections/node_pool.h>

namespace std {

generic<T>
struct ListNode {
    T                               value;
    ?&heap struct ListNode<T>       next;
    ?&heap struct ListNode<T>       prev;
};

generic<T>
struct List {
    ?&heap struct ListNode<T>       head;
    ?&heap struct ListNode<T>       tail;
    unsigned long                   len;
    struct NodePool<struct ListNode<T>> pool;

    static struct List<T> new();          // empty, no allocation

    int           push_front(T value);   // 0 on OOM
    int           push_back(T value);    // 0 on OOM
    ?T            pop_front();           // null if empty
    ?T            pop_back();            // null if empty
    ?&T           front() const;
    ?&T           back() const;
    unsigned long length() const;
    int           is_empty() const;
    void          clear();
    void          free();

    // cmp: <0, 0, >0 like strcmp; only == 0 matters here
    ?&T           find(const &T key, fn int(const &T, const &T) cmp) const;
    int           contains(const &T key, fn int(const &T, const &T) cmp) const;
    int           remove(const &T key, fn int(const &T, const &T) cmp);  // first match; 0 if none
    void          foreach(fn void(&T) func);
    void          reverse();

    ?&heap struct ListNode<T> node_(T value);
    void          unlink_(&heap struct ListNode<T> n);
};

generic<T>
struct List<T> List::new() {
    struct List<T> l;
    l.head = null;
    l.tail = null;
    l.len = 0UL;
    l.pool = NodePool::new();
    return l;
}

generic<T>
?&heap struct ListNode<T> List::node_(T value) {
    ?&heap struct ListNode<T> nq = self.pool.take();
    match (nq) {
        case some(n): { n.value = value; n.next = null; n.prev = null; }
        case null: { }
    }
    return nq;
}

// Detaches n from its neighbours (it stays allocated).
generic<T>
void List::unlink_(&heap struct ListNode<T> n) {
    match (n.prev) {
        case null:    self.head = n.next;
        case some(p): p.next = n.next;
    }
    match (n.next) {
        case null:    self.tail = n.prev;
        case some(q): q.prev = n.prev;
    }
    n.next = null;
    n.prev = null;
    self.len = self.len - 1UL;
}

generic<T>
int List::push_front(T value) {
    match (self.node_(value)) {
        case null: return 0;
        case some(n): {
            n.next = self.head;
            match (self.head) {
                case null:    self.tail = n;
                case some(h): h.prev = n;
            }
            self.head = n;
        }
    }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
int List::push_back(T value) {
    match (self.node_(value)) {
        case null: return 0;
        case some(n): {
            n.prev = self.tail;
            match (self.tail) {
                case null:    self.head = n;
                case some(t): t.next = n;
            }
            self.tail = n;
        }
    }
    self.len = self.len + 1UL;
    return 1;
}

generic<T>
?T List::pop_front() {
    match (self.head) {
        case null: return null;
        case some(n): {
            T v = n.value;
            self.unlink_(n);
            self.pool.give(n);
            return v;
        }
    }
    return null;
}

generic<T>
?T List::pop_back() {
    match (self.tail) {
        case null: return null;
        case some(n): {
            T v = n.value;
            self.unlink_(n);
            self.pool.give(n);
            return v;
        }
    }
    return null;
}

generic<T>
?&T List::front() const {
    match (self.head) {
        case null: return null;
        case some(n): unsafe { return (?&T)&n.value; }
    }
    return null;
}

generic<T>
?&T List::back() const {
    match (self.tail) {
        case null: return null;
        case some(n): unsafe { return (?&T)&n.value; }
    }
    return null;
}

generic<T>
unsigned long List::length() const { return self.len; }

generic<T>
int List::is_empty() const { return self.len == 0UL; }

generic<T>
void List::free() {
    self.pool.free_all();
    self.head = null;
    self.tail = null;
    self.len = 0UL;
}

generic<T>
void List::clear() { self.free(); }

generic<T>
?&T List::find(const &T key, fn int(const &T, const &T) cmp) const {
    ?&heap struct ListNode<T> cur = self.head;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                unsafe { if (cmp(&n.value, key) == 0) return (?&T)&n.value; }
                cur = n.next;
            }
            case null: { }
        }
    }
    return null;
}

generic<T>
int List::contains(const &T key, fn int(const &T, const &T) cmp) const {
    return !self.find(key, cmp).is_null();
}

generic<T>
int List::remove(const &T key, fn int(const &T, const &T) cmp) {
    ?&heap struct ListNode<T> cur = self.head;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                int hit = 0;
                unsafe { hit = cmp(&n.value, key) == 0; }
                if (hit) {
                    self.unlink_(n);
                    self.pool.give(n);
                    return 1;
                }
                cur = n.next;
            }
            case null: { }
        }
    }
    return 0;
}

generic<T>
void List::foreach(fn void(&T) func) {
    ?&heap struct ListNode<T> cur = self.head;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                unsafe { func(&n.value); }
                cur = n.next;
            }
            case null: { }
        }
    }
}

generic<T>
void List::reverse() {
    ?&heap struct ListNode<T> cur = self.head;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): {
                ?&heap struct ListNode<T> nx = n.next;
                n.next = n.prev;
                n.prev = nx;
                cur = nx;
            }
            case null: { }
        }
    }
    ?&heap struct ListNode<T> h = self.head;
    self.head = self.tail;
    self.tail = h;
}

} // namespace std
