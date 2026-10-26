// extracted from safec-docs book/ch05-understanding-regions.md (block 3)
int main() {
    struct Point local;
    local.x = 1.0; local.y = 2.0;
    &stack Point p = &local;
    double total = sum_point(p);   // &stack Point -> &Point: implicit
    printf("%f\n", total);
    return 0;
}
