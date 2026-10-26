// extracted from safec-docs reference/concurrency.md (block 4)
#include <std/collections/vec.h>
extern int printf(const char* fmt, ...);

struct WorkItem {
    const &Vec<int> data;       // shared for reading by both workers
    unsigned long start;
    unsigned long end;
    long long result;
};

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
    struct Vec<int> data = Vec::new();
    for (unsigned long i = 0UL; i < n; i = i + 1UL) data.push((int)i);

    struct WorkItem w1; w1.data = &data; w1.start = 0UL;     w1.end = n / 2UL; w1.result = 0LL;
    struct WorkItem w2; w2.data = &data; w2.start = n / 2UL; w2.end = n;       w2.result = 0LL;

    // Each worker gets its own WorkItem until it is joined; both read 'data'.
    long long h1 = spawn_scoped(sum_range, &w1);
    long long h2 = spawn_scoped(sum_range, &w2);
    join(h1);
    join(h2);

    printf("Total: %lld\n", w1.result + w2.result);
    data.drop();
    return 0;
}
