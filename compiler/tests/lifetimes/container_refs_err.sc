// SAFEC-ARGS: -I %root
// An element reference a container hands out ('at', 'get', 'front') points
// into its buffer: a call that may reallocate the buffer (Vec push/insert/
// reserve, a map insert that rehashes) or free it leaves it dangling.
#include <std/collections/vec.h>
#include <std/collections/map.h>
#include <std/collections/string.h>

int vec_at_push(&struct Vec<int> v) {
    &int r = v.at(0UL);
    v.push(1);
    return *r;                             // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after its buffer may have been reallocated
}

int vec_get_insert(&struct Vec<int> v) {
    match (v.get(0UL)) {
        case some(r): { v.insert(0UL, 7); return *r; }  // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after its buffer may have been reallocated
        case null: { }
    }
    return 0;
}

int vec_reserve_field(&struct Vec<struct String> v) {
    &struct String s = v.at(0UL);
    v.reserve(100UL);
    return (int)s.len;                     // EXPECT-ERROR: use of 's' (a reference into a &heap allocation) after its buffer may have been reallocated
}

int vec_free(struct Vec<int> v) {
    &int r = v.at(0UL);
    v.free();
    // EXPECT-ERROR: use of a reference borrowed from 'v' after 'v' was dropped
    return *r;                             // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after std::dealloc() freed it
}

int hashmap_get_insert(&struct HashMap<int, int> m) {
    match (m.get(1)) {
        case some(r): { m.insert(2, 2); return *r; }    // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after std::dealloc() freed it
        case null: { }
    }
    return 0;
}

struct Crate { struct StrMap<int> links; };
static struct StrMap<int> gLinks;

int strmap_field(&struct Crate c) {
    match (c.links.get("a")) {
        case some(r): { c.links.insert("b", 2); return *r; } // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after std::dealloc() freed it
        case null: { }
    }
    return 0;
}

int strmap_global() {
    match (gLinks.get("a")) {
        case some(r): { gLinks.insert("b", 2); return *r; }  // EXPECT-ERROR: use of 'r' (a reference into a &heap allocation) after std::dealloc() freed it
        case null: { }
    }
    return 0;
}
