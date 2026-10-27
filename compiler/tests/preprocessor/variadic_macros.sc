// SAFEC-ARGS: --compat-preprocessor
// __VA_ARGS__ is every argument from the '...' on, commas included (it was
// only the first one: 'F(1, 2, 3)' expanded to 'g(1)').
// EXEC
int g3(int a, int b, int c) { return a * 100 + b * 10 + c; }
int g2(int a, int b) { return a * 10 + b; }
#define ALL(...) g3(__VA_ARGS__)
#define REST(x, ...) g3(x, __VA_ARGS__)
#define STR(...) #__VA_ARGS__
#define NEST(...) ALL(__VA_ARGS__)

// CHECK-LABEL: define {{.*}}i32 @main(
int main() {
    if (ALL(1, 2, 3) != 123) return 1;
    if (REST(4, 5, 6) != 456) return 2;
    if (REST(7, g2(1, 2), 9) != 7 * 100 + 120 + 9) return 3;
    if (NEST(1, (2), 3) != 123) return 4;
    const char *s = STR(a, b);
    unsafe { if (s[0] != 'a' || s[1] != ',' || s[2] != ' ' || s[3] != 'b' || s[4] != 0) return 5; }
    return 0;
}
