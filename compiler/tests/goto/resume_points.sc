// A label right after a 'return' is still reachable through a goto: the
// stackless state machines std/async's ASYNC_BEGIN/AWAIT build resume
// there ('state = 1; return PENDING; R1: ...'). Codegen used to drop
// every statement after the return, leaving the label's block without a
// terminator (an LLVM verifier error). Jumping into a loop body is fine
// when no declaration in scope at the label is skipped.
struct St { int rp; int i; };

int step(&St s) {
    switch (s.rp) {
    case 0: break;
    case 1: goto R1;
    default: return -1;
    }
    s.i = 0;
    while (s.i < 3) {
        s.rp = 1;
        return 1;
R1:
        s.i = s.i + 1;
    }
    s.rp = 2;
    return 0;
}
// CHECK-LABEL: define {{.*}}@step(
// CHECK: ret i32

int run() {
    struct St s;
    s.rp = 0;
    s.i = 0;
    int n = 0;
    while (step(&s) == 1) n = n + 1;
    return n * 10 + s.i;
}
// run() == 33 (three suspensions, i == 3): checked by running it.

int main() { return run() == 33 ? 0 : 1; }
