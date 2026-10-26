// std's slice API and the slice forms it uses: functions returning '[]T',
// read-only '[]const T', '?[]T', '?fn', range-checked re-slicing, and the
// bundled std headers (str, fmt, mem, sys, collections) on slices.
#include <stdio.h>
#include <std/str.h>
#include <std/fmt.h>
#include <std/mem.h>
#include <std/sys.h>
#include <std/collections/slice.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

struct Buf { []int data; };
struct Ctx { int k; };

[]int middle([]int s) { return s[1..3]; }
?[]int maybe(int k, []int s) { if (k) return s; return null; }
int sum([]const int s) {
    int t = 0;
    for (unsigned long i = 0UL; i < s.len; i = i + 1UL) t = t + s[i];
    return t;
}
unsigned long count_bytes([]const unsigned char b) { return b.len; }
int scale(?&Ctx c, int x) {
    match (c) { case some(r): return x * r.k; case null: return x; }
    return x;
}
int apply(?fn int(?&Ctx, int) f, ?&Ctx c, int x) {
    match (f) { case some(g): return g(c, x); case null: return -1; }
    return 0;
}
int cmp_int(const &int a, const &int b) { return *a - *b; }

int main() {
    int a[6] = {5, 3, 9, 1, 7, 2};
    struct Buf b;
    b.data = a[0..6];
    []const int ro = middle(b.data);
    printf("%d %d %lu\n", sum(ro), sum(b.data), middle(a[..]).len);
    match (maybe(0, a[..])) { case some(s): printf("some %lu\n", s.len); case none: printf("none\n"); }
    match (maybe(1, a[2..])) { case some(s): printf("some %lu\n", s.len); case none: printf("none\n"); }

    struct Ctx c;
    c.k = 3;
    ?fn int(?&Ctx, int) none_ = null;
    printf("%d %d\n", apply(scale, (?&Ctx)&c, 4), apply(none_, (?&Ctx)&c, 4));

    char t[4] = {'a', 'b', 'c', 'd'};
    []char ct = t[0..4];
    printf("%lu\n", count_bytes(ct));

    char buf[8];
    int full = str_copy(buf[..], "hello");
    int cut = str_cat(buf[..], " world");
    printf("%s %d %d %lu\n", buf, full, cut, str_bytes("abc").len);

    char num[24];
    fmt_int(num[..], -1234L);
    printf("%s\n", num);
    fmt_hex(num[..], 255UL);
    printf("%s\n", num);

    int dst[3];
    unsigned long n = std::safe_memcpy(dst[..], a[3..]);
    printf("%lu %d %d %d %d\n", n, dst[0], dst[1], dst[2], std::safe_memcmp(dst[..], a[3..6]));
    std::safe_memset(dst[..], 0);
    printf("%d\n", sum(dst[..]));

    sys_qsort(a[..], cmp_int);
    printf("%d %d %d\n", a[0], a[3], a[5]);
    int key = 7;
    match (sys_bsearch(&key, a[..], cmp_int)) {
        case some(i): printf("at %lu\n", i);
        case none: printf("missing\n");
    }

    arr_reverse(a[..]);
    printf("%d %d\n", a[0], a[5]);
    struct Vec<int> v = Vec::from_slice(a[1..4]);
    printf("%lu %d\n", v.len, sum(v.as_slice()));
    v.free();

    struct String s = String::from_bytes(str_bytes("slices"));
    s.push_bytes(str_bytes("!"));
    printf("%s %lu\n", s.as_ptr(), s.as_bytes().len);
    s.free();
    return 0;
}
