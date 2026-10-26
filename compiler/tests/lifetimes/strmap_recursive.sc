// SAFEC-ARGS: -I %root
// A std StrMap insert (which may rehash, freeing the map's old arrays)
// after a recursive call doesn't dangle the Vec buffer next to it, whether
// the map is a struct field or a global — the recursive call's fresh Vec
// buffer and fresh map arrays are distinct objects.
#include <std/collections/map.h>
#include <std/collections/vec.h>
struct Crate { struct Vec<int> mods; struct StrMap<int> links; };
static struct StrMap<long long> gLinks;

long long ensure(&struct Crate c, int depth) {
    if (depth > 0) { ensure(c, depth - 1); }
    c.mods.push(depth);
    c.links.insert("k", depth);
    return (long long)*c.mods.at(0UL);
}

// The shape of safec-doc's ensure_module: the insert happens in the
// recursive call, the read after it.
static void add_link(const &char key, long long idx) { gLinks.insert(key, idx); }

long long ensure_global(&struct Crate c, int depth) {
    if (depth > 0) { ensure_global(c, depth - 1); }
    c.mods.push(depth);
    long long v = (long long)*c.mods.at(0UL);
    add_link("k", (long long)c.mods.len);
    return v;
}
