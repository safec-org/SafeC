// A call's result is a temporary that dies with its full expression: a
// fresh allocation returned by one side of a join ('?:', if/else, match,
// switch, '&&'/'||') doesn't carry the previous iteration's (freed) result
// of the other side into the next iteration.
void dealloc(?&heap float p);

// A minimal owning buffer: free() releases 'data' and nulls it.
struct Buf { ?&heap float data; unsigned long len; };
static struct Buf zeros(unsigned long n) {
    struct Buf b;
    &heap float d = new float;
    *d = (float)0.0;
    b.data = d;
    b.len = n;
    return b;
}
float Buf::first() const {
    match (self.data) {
        case some(d): return *d;
        case null: return (float)0.0;
    }
    return (float)0.0;
}
unsigned long Buf::length() const { return self.len; }
void Buf::free() { dealloc(self.data); self.data = null; }

static ?&heap float boxed(int x) { ?&heap float p = new float; return p; }

float ternary_in_loop() {
    int k = 0;
    float s = (float)0.0;
    while (k < 3) {
        struct Buf a = k == 1 ? zeros(4UL) : zeros(5UL);
        s = s + a.first();
        a.free();
        k = k + 1;
    }
    return s;
}

float if_else_in_loop() {
    float s = (float)0.0;
    for (int k = 0; k < 3; k = k + 1) {
        struct Buf a;
        if (k == 1) a = zeros(4UL); else a = zeros(5UL);
        s = s + a.first();
        a.free();
    }
    return s;
}

float nested_loops() {
    float s = (float)0.0;
    for (int i = 0; i < 2; i = i + 1) {
        for (int k = 0; k < 3; k = k + 1) {
            struct Buf a = k == i ? zeros(4UL) : zeros(5UL);
            s = s + a.first();
            a.free();
        }
    }
    return s;
}

float match_expr_in_loop() {
    float s = (float)0.0;
    for (int k = 0; k < 3; k = k + 1) {
        struct Buf a = match (k) { case 0: zeros(3UL), case 1: zeros(4UL), default: zeros(5UL) };
        s = s + a.first();
        a.free();
    }
    return s;
}

float switch_in_loop() {
    float s = (float)0.0;
    for (int k = 0; k < 3; k = k + 1) {
        struct Buf a;
        switch (k) {
        case 0: a = zeros(3UL); break;
        case 1: a = zeros(4UL); break;
        default: a = zeros(5UL); break;
        }
        s = s + a.first();
        a.free();
    }
    return s;
}

// Fresh scalar results joined by '?:' and consumed by '&&'.
int short_circuit_in_loop() {
    int n = 0;
    for (int k = 0; k < 3; k = k + 1) {
        ?&heap float p = k == 1 ? boxed(1) : boxed(2);
        if (!p.is_null() && k > 0) n = n + 1;
        dealloc(p);
    }
    return n;
}

// The loop condition is a full expression too.
int condition_temporaries() {
    int k = 0;
    int n = 0;
    do {
        struct Buf a = k == 0 ? zeros(2UL) : zeros(3UL);
        n = n + (int)a.length();
        a.free();
        k = k + 1;
    } while (k < 3 && zeros(1UL).length() > 0UL);
    return n;
}

// CHECK-LABEL: define {{.*}}@main(
int main() {
    return (int)(ternary_in_loop() + if_else_in_loop() + nested_loops() + match_expr_in_loop() +
                 switch_in_loop()) + short_circuit_in_loop() + condition_temporaries();
}
