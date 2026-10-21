#pragma once
// SafeC Standard Library — compile-time orderings
//
// An ordering is a type with 'int cmp(const &T a, const &T b) const'
// returning <0, 0 or >0 — the Order trait below. Containers and algorithms
// take the ordering as a type parameter bounded by Order
// (BST<K, V, C: Order>, vec_sort_by(&v, ord)), so a type without cmp is
// rejected where it is named, and after monomorphization each comparison is
// a direct call the optimizer inlines — no function pointer, no per-compare
// indirect call. FnOrder<T> wraps a runtime function pointer for orderings
// only known at run time; any struct of your own with a cmp method is an
// Order too.
//
//   struct BST<int, int, Less<int>> t = std::BST::new();     // natural order, inlined
//   std::vec_sort_by(&v, std::FnOrder::of(by_name));         // runtime comparator

namespace std {

// Stands for "the element type" in trait signatures here and in
// std/collections/traits.h: a trait fixes each method's name, static-ness
// and parameter count; the parameter types are each implementation's own.
typedef void* Elem_;

trait Order {
    // <0 if a sorts before b, 0 if they are equivalent, >0 if after.
    int cmp(const &Elem_ a, const &Elem_ b) const;
}

// Ascending by '<' / '>' (integers, floats, enums, pointers).
generic<T>
struct Less {
    int unused_;
    static struct Less<T> new();
    int cmp(const &T a, const &T b) const;
};

generic<T>
int Less::cmp(const &T a, const &T b) const { return *a < *b ? -1 : (*a > *b ? 1 : 0); }

// Descending.
generic<T>
struct Greater {
    int unused_;
    static struct Greater<T> new();
    int cmp(const &T a, const &T b) const;
};

generic<T>
int Greater::cmp(const &T a, const &T b) const { return *a > *b ? -1 : (*a < *b ? 1 : 0); }

// A comparator chosen at run time.
generic<T>
struct FnOrder {
    fn int(const &T, const &T) f;
    static struct FnOrder<T> of(fn int(const &T, const &T) f);
    int cmp(const &T a, const &T b) const;
};

generic<T>
int FnOrder::cmp(const &T a, const &T b) const { return self.f(a, b); }

generic<T>
struct Less<T> Less::new() { struct Less<T> o; o.unused_ = 0; return o; }

generic<T>
struct Greater<T> Greater::new() { struct Greater<T> o; o.unused_ = 0; return o; }

generic<T>
struct FnOrder<T> FnOrder::of(fn int(const &T, const &T) f) { struct FnOrder<T> o; o.f = f; return o; }

} // namespace std
