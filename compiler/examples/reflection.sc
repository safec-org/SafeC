extern int printf(const char *fmt, ...);

struct Point { double x; double y; double z; };

int main() {
    printf("alignof(int)=%lu\n", alignof(int));
    printf("alignof(double)=%lu\n", alignof(double));
    printf("fieldcount(Point)=%lu\n", fieldcount(Point));

    int x = 42;
    typeof(x) y = 100;
    printf("x=%d y=%d\n", x, y);

    static_assert(fieldcount(Point) == 3, "Point has 3 fields");
    return 0;
}
