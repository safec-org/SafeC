// extracted from safec-docs reference/polymorphism.md (block 1)
extern int printf(const char* fmt, ...);
struct Circle {
    double radius;
    double area() const;
};
double Circle::area() const { return 3.14159265 * self.radius * self.radius; }

struct Square {
    double side;
    double area() const;
};
double Square::area() const { return self.side * self.side; }

// A "shape" declaration -- name (area) and signature (takes nothing,
// returns double, self excluded) are the only things that matter. It is
// never called.
double area() { return 0.0; }

generic<T>
double describe_area(T obj) {
    return fn_eval(obj, area)(&obj);
}

int main() {
    struct Circle c; c.radius = 2.0;
    struct Square s; s.side = 3.0;

    double ca = describe_area(c);   // describe_area<Circle> resolves fn_eval to Circle::area
    double sa = describe_area(s);   // describe_area<Square> resolves fn_eval to Square::area
    printf("circle=%.4f square=%.4f\n", ca, sa);   // circle=12.5664 square=9.0000
    return 0;
}
