// extracted from safec-docs book/ch09-final-project.md (block 1)
extern int printf(const char* fmt, ...);

int main() {
    struct Store store;
    store.init();
    defer store.free();

    store.set("name", "SafeC");
    store.set("kind", "language");

    ?struct String name = store.get("name");
    match (name) {
        case none:    unsafe { printf("name: (not found)\n"); }
        case some(v): { unsafe { printf("name: %s\n", v.as_ptr()); } v.free(); }
    }

    int removed = store.remove("kind");
    unsafe { printf("removed kind: %d\n", removed); }

    ?struct String kind = store.get("kind");
    match (kind) {
        case none:    unsafe { printf("kind: (not found, as expected)\n"); }
        case some(v): { unsafe { printf("kind: %s\n", v.as_ptr()); } v.free(); }
    }

    return 0;
}
