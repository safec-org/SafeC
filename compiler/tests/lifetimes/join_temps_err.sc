// The same joins of fresh call results still catch a genuine use after
// free: through a copy of the joined Buf (Buf::free nulls only its own
// 'data'), a double free through such a copy, and an alias kept from an
// earlier iteration.
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

float use_after_free_in_loop() {
    int k = 0;
    float s = (float)0.0;
    while (k < 3) {
        struct Buf a = k == 1 ? zeros(4UL) : zeros(5UL);
        struct Buf b = a;
        a.free();
        s = s + b.first();               // EXPECT-ERROR: call to 'Buf_first' reads a &heap reference after std::dealloc() freed it
        k = k + 1;
    }
    return s;
}

float freed_twice() {
    float s = (float)0.0;
    for (int k = 0; k < 3; k = k + 1) {
        struct Buf a = match (k) { case 0: zeros(3UL), default: zeros(5UL) };
        struct Buf b = a;
        a.free();
        b.free();                         // EXPECT-ERROR: call to 'Buf_free' frees memory after std::dealloc() freed it
    }
    return s;
}

// An alias kept from the previous iteration, whose allocation that
// iteration freed.
int stale_alias() {
    ?&heap float keep = null;
    int n = 0;
    for (int k = 0; k < 3; k = k + 1) {
        ?&heap float p = k == 1 ? boxed(1) : boxed(2);
        if (k > 0 && !keep.is_null()) n = n + 1;   // EXPECT-ERROR: use of 'keep' (&heap reference) after std::dealloc() freed it
        keep = p;
        dealloc(p);
    }
    return n;
}

int main() { return 0; }
