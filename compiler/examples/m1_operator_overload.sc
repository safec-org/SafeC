// M1: Operator Overloading demo
extern int printf(const char *fmt, ...);

struct Vec2 {
    double x;
    double y;
    Vec2 operator+(Vec2 other) const;
    Vec2 operator-(Vec2 other) const;
    double dot(Vec2 other) const;
};

Vec2 Vec2::operator+(Vec2 other) const {
    Vec2 result;
    result.x = self.x + other.x;
    result.y = self.y + other.y;
    return result;
}

Vec2 Vec2::operator-(Vec2 other) const {
    Vec2 result;
    result.x = self.x - other.x;
    result.y = self.y - other.y;
    return result;
}

double Vec2::dot(Vec2 other) const {
    return self.x * other.x + self.y * other.y;
}

int main() {
    Vec2 a;
    a.x = 3.0;
    a.y = 4.0;
    Vec2 b;
    b.x = 1.0;
    b.y = 2.0;
    Vec2 c = a + b;
    printf("c = (%f, %f)\n", c.x, c.y);
    double d = a.dot(b);
    printf("a.dot(b) = %f\n", d);
    return 0;
}
