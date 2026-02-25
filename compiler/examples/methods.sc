// SafeC method example — demonstrates struct methods per OBJECT.md
extern int printf(const char* fmt, ...);

// ── Simple point struct with methods ─────────────────────────────────────────
struct Point {
    double x;
    double y;
    double length() const;
    double dot(double ox, double oy) const;
    void   scale(double s);
};

double Point::length() const {
    return self.x * self.x + self.y * self.y; // simplified: squared length
}

double Point::dot(double ox, double oy) const {
    return self.x * ox + self.y * oy;
}

void Point::scale(double s) {
    self.x = self.x * s;
    self.y = self.y * s;
}

// ── Counter with methods ──────────────────────────────────────────────────────
struct Counter {
    int value;
    void   reset();
    void   increment();
    void   add(int n);
    int    get() const;
};

void Counter::reset() {
    self.value = 0;
}

void Counter::increment() {
    self.value = self.value + 1;
}

void Counter::add(int n) {
    self.value = self.value + n;
}

int Counter::get() const {
    return self.value;
}

int main() {
    // Test Point methods
    struct Point p;
    p.x = 3.0;
    p.y = 4.0;
    double len2 = p.length(); // 9.0 + 16.0 = 25.0
    unsafe { printf("Point(3,4) length^2 = %g\n", len2); }

    double d = p.dot(1.0, 0.0); // 3*1 + 4*0 = 3.0
    unsafe { printf("dot((3,4),(1,0)) = %g\n", d); }

    p.scale(2.0);
    unsafe { printf("After scale(2): x=%g y=%g\n", p.x, p.y); }

    // Test Counter methods
    struct Counter c;
    c.reset();
    c.increment();
    c.increment();
    c.add(5);
    int val = c.get(); // 0 + 1 + 1 + 5 = 7
    unsafe { printf("Counter value: %d\n", val); }

    return 0;
}
