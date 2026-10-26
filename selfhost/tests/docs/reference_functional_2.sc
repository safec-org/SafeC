// extracted from safec-docs reference/functional.md (block 2)
extern int printf(const char* fmt, ...);
?int safe_div(int a, int b) {
    if (b == 0) { return null; }   // the empty case
    return a / b;                  // implicit T -> ?T wrap
}

int main() {
    int r1 = match (safe_div(10, 2)) {
        case none:    -1,
        case some(v): v,
    };
    int r2 = match (safe_div(10, 0)) {
        case none:    -1,
        case some(v): v,
    };
    printf("10/2=%d 10/0=%d\n", r1, r2);   // 10/2=5 10/0=-1
    return 0;
}
