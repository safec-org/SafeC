#include <std/io.h>
#include "geo/point.h"

int main() {
    struct Point p = Point::new(3, 4);
    if (p.norm2() != 25LL) { std::println("norm2 wrong"); return 1; }
    struct String s = p.describe();
    int ok = s.eq_cstr("(3, 4)");
    s.free();
    if (!ok) { std::println("describe wrong"); return 1; }
    if (geo::shape_area(SHAPE_SQUARE, 3) != 9LL) { std::println("area wrong"); return 1; }
    std::println("geo ok");
    return 0;
}
