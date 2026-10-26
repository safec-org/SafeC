// extracted from safec-docs book/ch05-understanding-regions.md (block 1)
extern int printf(const char* fmt, ...);

void print_x(&stack int ref) {
    unsafe { printf("%d\n", *ref); }
}

int main() {
    int v = 7;
    print_x(&v);   // &v is a &stack int -- valid for main()'s whole scope
    return 0;
}
