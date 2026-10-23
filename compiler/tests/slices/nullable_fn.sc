// '?fn R(Args)' is a nullable function reference: it takes null, a function
// name or a 'fn' value, and is read with match / is_null() like any '?&T'.
// A generic parameter of that type takes a bare null without binding T.
struct Ctx { int v; };
int twice(?&Ctx c, int x) { return x * 2; }
generic<T> int call_or(?fn int(?&T, int) f, ?&T c, int x) {
    match (f) {
        case some(g): return g(c, x);
        case null:    return -1;
    }
    return 0;
}
// CHECK: define {{.*}}@main(
int main() {
    struct Ctx c;
    c.v = 1;
    ?fn int(?&Ctx, int) none_ = null;
    ?fn int(?&Ctx, int) some_ = twice;
    if (!none_.is_null()) return 1;
    return call_or(some_, (?&Ctx)&c, 3) - 6 + call_or(none_, (?&Ctx)&c, 3) + 1;
}
