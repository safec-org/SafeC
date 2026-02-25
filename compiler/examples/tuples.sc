// tuples.sc — M3: Tuple lowering
extern int printf(const char *fmt, ...);

int main() {
    tuple(int, double) p = (42, 3.14);
    int x = p.0;
    double y = p.1;
    printf("p.0 = %d, p.1 = %f\n", x, y);

    tuple(int, int, int) rgb = (255, 128, 0);
    printf("rgb = (%d, %d, %d)\n", rgb.0, rgb.1, rgb.2);

    return 0;
}
