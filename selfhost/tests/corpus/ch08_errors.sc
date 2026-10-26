#include <stdio.h>
#include <stdlib.h>
?int parse_positive(const &char s) {
    int v;
    unsafe { v = atoi(s); }
    if (v <= 0) { return null; }
    return v;
}
?int double_positive(const &char s) {
    int v = try parse_positive(s);
    return v * 2;
}
void demo() {
    printf("start\n");
    defer printf("first deferred\n");
    defer printf("second deferred\n");
    printf("middle\n");
}
?int risky_step(int fail_at) {
    if (fail_at != 0) { return null; }
    return 0;
}
?int process(int fail_at) {
    printf("processing\n");
    errdefer printf("rolling back\n");
    int r = try risky_step(fail_at);
    return 42 + r;
}
void show(?int v) {
    match (v) {
        case none:    printf("none\n");
        case some(x): printf("some(%d)\n", x);
    }
}
int main() {
    match (parse_positive("21")) {
        case none:    printf("nothing\n");
        case some(x): printf("got %d\n", x);
    }
    show(double_positive("21"));
    show(double_positive("-5"));
    demo();
    show(process(0));
    show(process(1));
    ?int e = null;
    printf("%d %d\n", (int)e.is_none(), e.default(7));
    return 0;
}
