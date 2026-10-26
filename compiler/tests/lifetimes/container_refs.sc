// SAFEC-ARGS: -I %root
// Element references stay valid across everything that doesn't reallocate
// or free their own container's buffer.
#include <std/collections/vec.h>
#include <std/collections/map.h>
#include <std/collections/string.h>

struct Item { struct String name; struct Vec<int> vals; };

// Re-fetched after the push.
int refetch(&struct Vec<int> v) {
    v.push(1);
    &int r = v.at(0UL);
    return *r;
}

// Non-growing calls on the same container, and growing another one.
int other_calls(&struct Vec<int> v, &struct Vec<int> w) {
    &int r = v.at(0UL);
    v.set(1UL, 5);
    unsigned long n = v.length();
    w.push(*r);
    return *r + (int)n;
}

// Growing a Vec stored inside an element doesn't move the element.
int inner_push(&struct Vec<struct Item> items) {
    &struct Item e = items.at(0UL);
    e.vals.push(1);
    e.vals.push(2);
    e.name.push("x");
    return (int)e.vals.len;
}

// Freeing each element's own memory, one element at a time (taking it
// out of the element: the Vec drops what is left).
void free_all(&struct Vec<struct Item> items) {
    for (unsigned long i = 0UL; i < items.len; i = i + 1UL) {
        &struct Item e = items.at(i);
        std::replace(&e.name, String::new()).drop();
        struct Vec<int> none = Vec::new();
        std::replace(&e.vals, none).drop();
    }
    items.clear();
}

// A map lookup used before the next insert.
int map_lookup(&struct StrMap<int> m) {
    int v = 0;
    match (m.get("a")) { case some(r): { v = *r; } case null: { } }
    m.insert("b", v);
    match (m.get("b")) { case some(r): { return *r; } case null: { } }
    return v;
}
