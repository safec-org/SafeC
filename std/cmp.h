#pragma once
// SafeC Standard Library — compile-time orderings
//
// An ordering is a type with 'int cmp(const &T a, const &T b) const'
// returning <0, 0 or >0. Containers and algorithms take the ordering as a
// type parameter (BST<K, V, Less<K>>, vec_sort_by(&v, ord)), so after
// monomorphization each comparison is a direct call the optimizer inlines —
// no function pointer, no per-compare indirect call. FnOrder<T> wraps a
// runtime function pointer for orderings only known at run time.
//
//   struct BST<int, int, Less<int>> t = std::bst_new();     // natural order, inlined
//   std::vec_sort_by(&v, std::order_fn(by_name));            // runtime comparator

namespace std {

// Ascending by '<' / '>' (integers, floats, enums, pointers).
generic<T>
struct Less {
    int unused_;
    int cmp(const &T a, const &T b) const;
};

generic<T>
int Less::cmp(const &T a, const &T b) const { return *a < *b ? -1 : (*a > *b ? 1 : 0); }

// Descending.
generic<T>
struct Greater {
    int unused_;
    int cmp(const &T a, const &T b) const;
};

generic<T>
int Greater::cmp(const &T a, const &T b) const { return *a > *b ? -1 : (*a < *b ? 1 : 0); }

// A comparator chosen at run time.
generic<T>
struct FnOrder {
    fn int(const &T, const &T) f;
    int cmp(const &T a, const &T b) const;
};

generic<T>
int FnOrder::cmp(const &T a, const &T b) const { return self.f(a, b); }

generic<T>
struct Less<T> order_less() { struct Less<T> o; o.unused_ = 0; return o; }

generic<T>
struct Greater<T> order_greater() { struct Greater<T> o; o.unused_ = 0; return o; }

generic<T>
struct FnOrder<T> order_fn(fn int(const &T, const &T) f) { struct FnOrder<T> o; o.f = f; return o; }

} // namespace std
