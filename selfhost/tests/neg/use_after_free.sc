// use of a heap reference after std::dealloc
// expect-error: use of 'p' (&heap reference) after std::dealloc() freed it
#include <std/mem.h>
int main() {
    &heap int p = new int;
    *p = 1;
    std::dealloc(p);
    return *p;
}
