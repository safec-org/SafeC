#include <stdio.h>
namespace geo {
    int square(int x) { return x * x; }
    int counter = 41;
    namespace inner { int triple(int x) { return x * 3; } }
    struct Pt { int x; };
}
newtype UserId = int;
typedef unsigned long u64;
typedef struct { int a; int b; } Pair;
int main() {
    int a = geo::square(6);
    int b = geo::inner::triple(4);
    geo::counter = geo::counter + 1;
    int c = square(6);
    struct Pt p;
    p.x = 3;
    UserId id = (UserId)42;
    u64 big = 5UL;
    Pair pr;
    pr.a = 1; pr.b = 2;
    printf("%d %d %d %d %d %d %lu %d\n", a, b, geo::counter, c, p.x, (int)id, big, pr.a + pr.b);
    return 0;
}
