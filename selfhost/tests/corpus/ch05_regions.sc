extern int printf(const char* fmt, ...);
region Pool { capacity: 1024 }
struct Point { double x; double y; };
static int counter = 0;
&static int get_counter() { return &counter; }
void print_x(&stack int ref) { unsafe { printf("%d\n", *ref); } }
double sum_point(&Point p) { return p.x + p.y; }
void nullable_demo(?&stack struct Point maybe) {
    match (maybe) {
        case null:    printf("no point\n");
        case some(p): printf("point: %f\n", p.x);
    }
}
void arena_demo() {
    &arena<Pool> struct Point p = new<Pool> struct Point;
    p->x = 1.0;
    p->y = 2.0;
    printf("arena point: %f %f\n", p->x, p->y);
    unsigned long mark = arena_mark<Pool>();
    &arena<Pool> int scratch = new<Pool> int;
    *scratch = 99;
    printf("scratch %d\n", *scratch);
    arena_free_to<Pool>(mark);
    printf("still %f\n", p->x);
    arena_reset<Pool>();
}
int main() {
    int v = 7;
    print_x(&v);
    &static int c = get_counter();
    *c = 5;
    printf("counter %d\n", counter);
    struct Point local;
    local.x = 1.0; local.y = 2.0;
    &stack Point p = &local;
    printf("%f\n", sum_point(p));
    nullable_demo(&local);
    nullable_demo(null);
    arena_demo();
    int x = 42;
    &stack const int a = &x;
    &stack const int b = &x;
    printf("borrows: %d %d\n", *a, *b);
    return 0;
}
