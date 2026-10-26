// extracted from safec-docs reference/polymorphism.md (block 2)
extern int printf(const char* fmt, ...);

struct Circle { double radius; };
struct Square { double side; };

static double circle_area(&Circle c) { return 3.14159265 * c.radius * c.radius; }
static double square_area(&Square s) { return s.side * s.side; }

struct Shape {
    ?&void obj;                 // erased; only ever written by make_shape
    fn double(&void) area;      // really fn double(&T), same T as obj
    double eval_area();
};

generic<T>
struct Shape make_shape(&T obj, fn double(&T) area) {
    struct Shape sh;
    unsafe { sh.obj = (?&void)obj; sh.area = (fn double(&void))area; }   // the one erasure, here only
    return sh;
}

double Shape::eval_area() {
    match (self.obj) {
        case some(o): return self.area(o);
        case null:    return 0.0;
    }
    return 0.0;
}

static struct Circle g_c;
static struct Square g_sq;

int main() {
    g_c.radius = 2.0;
    g_sq.side = 3.0;
    struct Shape shapes[2];
    unsafe {
        shapes[0] = make_shape(&g_c, circle_area);    // make_shape(&g_c, square_area) won't compile
        shapes[1] = make_shape(&g_sq, square_area);
    }
    int i = 0;
    while (i < 2) {
        double a = 0.0;
        unsafe { a = shapes[i].eval_area(); printf("shape[%d] area=%.4f\n", i, a); }   // 12.5664, then 9.0000
        i = i + 1;
    }
    return 0;
}
