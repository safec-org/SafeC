// static_assert failure
// expect-error: static_assert failed: int is 8 bytes
static_assert(sizeof(int) == 8, "int is 8 bytes");
int main() { return 0; }
