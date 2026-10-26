#include <stdio.h>
?int find_first([]int arr, int target) {
    long i = 0L;
    while (i < (long)arr.len) {
        if (arr[i] == target) { return (int)i; }
        i = i + 1L;
    }
    return null;
}
tuple(int, double) divmod(int a, int b) { return (a / b, (double)(a % b)); }
int sum(tuple(int, int) t) { return t.0 + t.1; }
int total([]int s) {
    int t = 0;
    for (unsigned long i = 0UL; i < s.len; i = i + 1UL) { t = t + s[i]; }
    return t;
}
int main() {
    int arr[5] = {10, 20, 30, 40, 50};
    []int s = arr[1..4];
    printf("%lu %d %d\n", s.len, s[0], s[2]);
    printf("%d\n", total(arr[0..5]));
    printf("%d\n", total(s));
    match (find_first(arr[0..5], 40)) {
        case some(i): printf("found at %d\n", i);
        case none: printf("missing\n");
    }
    tuple(int, double) pair = (42, 3.14);
    printf("%d %.2f\n", pair.0, pair.1);
    tuple(int, double) dm = divmod(17, 5);
    printf("%d %.1f\n", dm.0, dm.1);
    printf("%d\n", sum((3, 4)));
    tuple(long long, int) q = (1, 2);
    printf("%lld %d\n", q.0, q.1);
    return 0;
}
