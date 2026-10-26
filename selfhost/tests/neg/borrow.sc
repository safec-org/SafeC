// a mutable borrow excludes other borrows
// expect-error: cannot borrow 'x' as mutable: already borrowed as mutable
int main() {
    int x = 1;
    &stack int a = &x;
    &stack int b = &x;
    *a = 2;
    return *b;
}
