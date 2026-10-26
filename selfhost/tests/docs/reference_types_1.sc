// extracted from safec-docs reference/types.md (block 1)
extern int printf(const char* fmt, ...);
union Result {
    int ok;
    int err;
}

void handle(union Result r) {
    match (r) {
        case .ok(v):  printf("ok: %d\n", v);
        case .err(e): printf("err: %d\n", e);
        default:      printf("unreachable\n");
    }
}

int main() {
    union Result a = Result.ok(42);
    union Result b = Result.err(-1);
    handle(a);   // ok: 42
    handle(b);   // err: -1
    return 0;
}
