// M2: Tuple types demo
extern int printf(const char *fmt, ...);

tuple(int, double) make_pair(int a, double b) {
    return (a, b);
}

int main() {
    tuple(int, double) t = (42, 3.14);
    printf("t.0 = %d, t.1 = %f\n", t.0, t.1);
    
    tuple(int, double) p = make_pair(7, 2.71);
    printf("p.0 = %d, p.1 = %f\n", p.0, p.1);
    return 0;
}
