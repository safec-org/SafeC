// extracted from safec-docs reference/ffi.md (block 2)
struct Widget { int value; }

// Stand-in for 'extern int c_register_callback(struct Widget*);' — a SafeC
// function with the same shape takes the reference, C would get the address.
int c_register_callback(?&Widget w) {
    match (w) {
        case some(x): return x.value;
        case null:    return 0;
    }
    return 0;
}

int main() {
    struct Widget local;
    local.value = 42;

    &stack Widget sref = &local;
    ?&Widget wref = sref;         // &stack Widget -> ?&Widget: implicit, no unsafe
    c_register_callback(wref);    // ?&Widget -> struct Widget*: implicit, no unsafe
    return 0;
}
