// methods.sc — struct methods with static dispatch
//
// Demonstrates:
//   - Method declarations inside struct body
//   - Method definitions using Struct::method() qualified syntax
//   - Implicit 'self' receiver (const and mutable)
//   - Method calls using dot notation: x.m(args)
//
// extern signatures use raw C types (see README §9.1).

extern int printf(const char* fmt, ...);

// ── Point: geometry methods ───────────────────────────────────────────────────

struct Point {
    double x;
    double y;
    double length() const;
    double dot(double ox, double oy) const;
    void   scale(double s);
};

double Point::length() const {
    return self.x * self.x + self.y * self.y; // squared length
}

double Point::dot(double ox, double oy) const {
    return self.x * ox + self.y * oy;
}

void Point::scale(double s) {
    self.x = self.x * s;
    self.y = self.y * s;
}

// ── Counter: mutable state via methods ────────────────────────────────────────

struct Counter {
    int value;
    void reset();
    void increment();
    void add(int n);
    int  get() const;
};

void Counter::reset()      { self.value = 0; }
void Counter::increment()  { self.value = self.value + 1; }
void Counter::add(int n)   { self.value = self.value + n; }
int  Counter::get() const  { return self.value; }

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // Point methods
    Point p;
    p.x = 3.0;
    p.y = 4.0;

    double len2 = p.length();        // 9 + 16 = 25
    printf("Point(3,4) length^2  = %g\n", len2);

    double d = p.dot(1.0, 0.0);     // 3*1 + 4*0 = 3
    printf("dot((3,4),(1,0))      = %g\n", d);

    p.scale(2.0);
    printf("After scale(2): (%g, %g)\n", p.x, p.y);

    // Counter methods
    Counter c;
    c.reset();
    c.increment();
    c.increment();
    c.add(5);
    int val = c.get();               // 0 + 1 + 1 + 5 = 7
    printf("Counter value          = %d\n", val);

    return 0;
}
