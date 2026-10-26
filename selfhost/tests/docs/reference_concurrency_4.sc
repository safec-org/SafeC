// extracted from safec-docs reference/concurrency.md (block 4)
#include <std/collections/slice.h>
extern int printf(const char* fmt, ...);

struct WorkItem {
    struct Slice<int> data;
    unsigned long start;
    unsigned long end;
    long long result;
};

// Shared with the workers, so it lives in static storage: spawn only
// accepts references that can't dangle while the thread runs.
static struct WorkItem w1;
static struct WorkItem w2;

?&WorkItem sum_range(?&WorkItem arg) {
    match (arg) {
        case some(w): {
            long long sum = 0LL;
            for (unsigned long i = w.start; i < w.end; i = i + 1UL) {
                sum = sum + (long long)*w.data.at(i);   // bounds-checked
            }
            w.result = sum;
        }
        case null: { }
    }
    return null;
}

int main() {
    unsigned long n = 10000UL;
    struct Slice<int> data = std::Slice::alloc(n);      // heap, runtime-sized
    for (unsigned long i = 0UL; i < n; i = i + 1UL) data.set(i, (int)i);

    w1.data = data; w1.start = 0UL;    w1.end = n / 2UL; w1.result = 0LL;
    w2.data = data; w2.start = n / 2UL; w2.end = n;      w2.result = 0LL;

    long long h1 = spawn(sum_range, &w1);
    long long h2 = spawn(sum_range, &w2);
    join(h1);
    join(h2);

    printf("Total: %lld\n", w1.result + w2.result);
    data.free();
    return 0;
}
