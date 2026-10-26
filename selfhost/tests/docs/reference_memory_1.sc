// extracted from safec-docs reference/memory.md (block 1)
#include <std/collections/slice.h>
struct Sample { int v; };
region Frame { capacity: 4096 }

static int g_table[16];                      // static: size fixed at compile time

void fill(struct Slice<int> s, int v) {
    for (unsigned long i = 0UL; i < s.length(); i = i + 1UL) s.set(i, v);
}

int arrays(unsigned long n) {
    int local[8];                            // stack: size fixed at compile time
    // int bad[n];                           // error: 'n' is not a compile-time constant
    &heap int h = new int[n];                // heap: runtime size
    &arena<Frame> int a = new<Frame> int[n]; // arena: runtime size, within Frame's 4096 bytes

    fill(std::Slice::of(&local[0], 8UL), 1);  // view each one through a bounds-checked Slice
    fill(std::Slice::of(h, n), 2);
    fill(std::Slice::of(a, n), 3);
    int t = local[7] + *std::Slice::of(h, n).at(n - 1UL) + *std::Slice::of(a, n).at(0UL) + g_table[0];

    std::dealloc(h);
    arena_reset<Frame>();
    return t;                                // 1 + 2 + 3 + 0
}

// Arrays of references follow the same rules.
int count_live(struct Slice<&Sample> items) { return (int)items.length(); }

int main() {
    struct Sample a; a.v = 1;
    struct Sample b; b.v = 2;
    &Sample fixed[2] = { &a, &b };           // stack array of references
    int c = count_live(std::Slice::of(&fixed[0], 2UL));
    return arrays(4UL) == 6 && c == 2 ? 0 : 1;
}
