// a stack reference can't escape through the return value
// expect-error: cannot return '&stack int': stack reference escapes function scope
&stack int leak() {
    int x = 5;
    return &x;
}
int main() { return *leak(); }
