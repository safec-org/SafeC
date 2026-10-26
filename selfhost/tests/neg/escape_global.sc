// a stack reference can't be stored in a global
// expect-error: cannot assign '&stack int' to variable in outer scope: reference would escape
?&static int g = null;
int main() {
    int x = 1;
    g = &x;
    return 0;
}
