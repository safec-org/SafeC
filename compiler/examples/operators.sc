// operators.sc — operator overloading + generic trait enforcement
//
// Demonstrates:
//   - Struct operator overloading (Vec2 +, ==)
//   - Generic functions that work with both built-in and struct types
//   - Operator overloaded structs used through generics (Counter + generic add_two)
//
// extern signatures use raw C types (see README §9.1).

extern int printf(const char *fmt, ...);

// ── Vec2: overloaded + and == ─────────────────────────────────────────────────

struct Vec2 {
    double x;
    double y;
    Vec2 operator+(Vec2 other) const;
    Vec2 operator-(Vec2 other) const;
    bool operator==(Vec2 other) const;
    double dot(Vec2 other) const;
};

Vec2 Vec2::operator+(Vec2 other) const {
    Vec2 r;
    r.x = self.x + other.x;
    r.y = self.y + other.y;
    return r;
}

Vec2 Vec2::operator-(Vec2 other) const {
    Vec2 r;
    r.x = self.x - other.x;
    r.y = self.y - other.y;
    return r;
}

bool Vec2::operator==(Vec2 other) const {
    return self.x == other.x && self.y == other.y;
}

double Vec2::dot(Vec2 other) const {
    return self.x * other.x + self.y * other.y;
}

// ── Counter: generic-compatible struct with operator+ ─────────────────────────

struct Counter {
    int value;
    Counter operator+(Counter other) const;
    Counter operator-(Counter other) const;
};

Counter Counter::operator+(Counter other) const {
    Counter r;
    r.value = self.value + other.value;
    return r;
}

Counter Counter::operator-(Counter other) const {
    Counter r;
    r.value = self.value - other.value;
    return r;
}

// ── Generic sum: works with int, double, and any overloaded struct ─────────────

generic<T>
T sum3(T a, T b, T c) {
    T tmp = a + b;
    return tmp + c;
}

generic<T>
T add_two(T a, T b) {
    return a + b;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // Vec2 operator overloads
    Vec2 a; a.x = 1.0; a.y = 2.0;
    Vec2 b; b.x = 3.0; b.y = 4.0;

    Vec2 c = a + b;
    printf("Vec2 + Vec2  = (%f, %f)\n", c.x, c.y);

    Vec2 d = b - a;
    printf("Vec2 - Vec2  = (%f, %f)\n", d.x, d.y);

    int eq = (a == b) ? 1 : 0;
    printf("a == b       = %d\n", eq);

    double dp = a.dot(b);
    printf("a.dot(b)     = %f\n", dp);

    // Generics: monomorphized for int, double
    int   si = sum3(1, 2, 3);
    double sd = sum3(1.0, 2.0, 3.0);
    printf("sum3(1,2,3)         = %d\n", si);
    printf("sum3(1.0,2.0,3.0)   = %f\n", sd);

    // Generics: monomorphized for Counter (has operator+)
    Counter x; x.value = 10;
    Counter y; y.value = 5;
    Counter z = add_two(x, y);
    printf("add_two(10, 5).value = %d\n", z.value);

    return 0;
}
