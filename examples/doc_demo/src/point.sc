#include "geo/point.h"
namespace geo {
struct Point Point::new(int x, int y) { struct Point p; p.x = x; p.y = y; return p; }
long long Point::norm2() const { long long x = (long long)self.x; long long y = (long long)self.y; return x * x + y * y; }
struct String Point::describe() const { struct String s = String::from("("); s.push_int((long long)self.x); s.push(", "); s.push_int((long long)self.y); s.push(")"); return s; }
long long shape_area(enum ShapeKind kind, int s) { long long n = (long long)s; if (kind == SHAPE_SQUARE) { return n * n; } return n * n * 314LL / 100LL; }
}
