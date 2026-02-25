// operators.sc — M1: Operator overloading + trait enforcement
extern int printf(const char *fmt, ...);

struct Vec2 {
    double x; double y;
    Vec2 operator+(Vec2 other) const;
    bool operator==(Vec2 other) const;
};

Vec2 Vec2::operator+(Vec2 other) const {
    Vec2 r;
    r.x = self.x + other.x;
    r.y = self.y + other.y;
    return r;
}

bool Vec2::operator==(Vec2 other) const {
    return self.x == other.x && self.y == other.y;
}

generic<T>
T sum3(T a, T b, T c) {
    T tmp = a + b;
    return tmp + c;
}

int main() {
    Vec2 a; a.x = 1.0; a.y = 2.0;
    Vec2 b; b.x = 3.0; b.y = 4.0;
    Vec2 c = a + b;
    printf("Vec2 + Vec2 = (%f, %f)\n", c.x, c.y);

    int eq = (a == b) ? 1 : 0;
    printf("a == b: %d\n", eq);

    int si = sum3(1, 2, 3);
    printf("sum3(1,2,3) = %d\n", si);

    double sd = sum3(1.0, 2.0, 3.0);
    printf("sum3(1.0,2.0,3.0) = %f\n", sd);

    return 0;
}
