#include <stdio.h>
struct Point {
    double x;
    double y;
    double length() const;
    void scale(double s);
};
double Point::length() const { return self.x * self.x + self.y * self.y; }
void Point::scale(double s) { self.x = self.x * s; self.y = self.y * s; }
struct Vec2 {
    double x;
    double y;
    Vec2 operator+(Vec2 other) const;
    Vec2 operator-(Vec2 other) const;
    Vec2 operator*(double s) const;
    bool operator==(Vec2 o) const;
};
Vec2 Vec2::operator+(Vec2 other) const { Vec2 r; r.x = self.x + other.x; r.y = self.y + other.y; return r; }
Vec2 Vec2::operator-(Vec2 other) const { Vec2 r; r.x = self.x - other.x; r.y = self.y - other.y; return r; }
Vec2 Vec2::operator*(double s) const { Vec2 r; r.x = self.x * s; r.y = self.y * s; return r; }
bool Vec2::operator==(Vec2 o) const { return self.x == o.x && self.y == o.y; }
struct Counter {
    int n;
    static struct Counter make(int start);
    void bump();
    int get() const;
};
struct Counter Counter::make(int start) { struct Counter c; c.n = start; return c; }
void Counter::bump() { self.n = self.n + 1; }
int Counter::get() const { return self.n; }
int main() {
    struct Point p = {3.0, 4.0};
    double len = p.length();
    p.scale(2.0);
    printf("%.1f %.1f %.1f\n", len, p.x, p.y);
    struct Vec2 a = {1.0, 2.0};
    struct Vec2 b = {3.0, 4.0};
    struct Vec2 c = a + b;
    struct Vec2 d = (c - a) * 2.0;
    printf("%.1f %.1f %.1f %.1f %d\n", c.x, c.y, d.x, d.y, (int)(a + b == c));
    struct Counter k = Counter::make(10);
    k.bump(); k.bump();
    printf("%d\n", k.get());
    Point q = p;
    q.x = 99.0;
    printf("%.1f %.1f\n", p.x, q.x);
    return 0;
}
