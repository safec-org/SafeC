// extracted from safec-docs book/ch05-understanding-regions.md (block 2)
region Pool { capacity: 1024 }
int main() {
    &arena<Pool> int p = new<Pool> int;
    arena_reset<Pool>();
    *p = 42;
    return 0;
}
