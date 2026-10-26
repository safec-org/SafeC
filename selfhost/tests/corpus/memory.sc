#include <stdio.h>
#include <stdlib.h>
struct Node { int value; ?&heap struct Node next; };
int main() {
    unsafe {
        int* raw = (int*)malloc(10UL * sizeof(int));
        for (int i = 0; i < 10; i = i + 1) { raw[i] = i * i; }
        int s = 0;
        for (int i = 0; i < 10; i = i + 1) { s = s + raw[i]; }
        printf("%d\n", s);
        raw = (int*)realloc((void*)raw, 20UL * sizeof(int));
        raw[19] = 7;
        printf("%d %d\n", raw[9], raw[19]);
        free((void*)raw);
    }
    &heap int p = new int;
    *p = 42;
    printf("heap: %d\n", *p);
    &heap struct Node a = new struct Node;
    &heap struct Node b = new struct Node;
    a.value = 1;
    b.value = 2;
    a.next = b;
    int total = 0;
    ?&heap struct Node cur = a;
    while (!cur.is_null()) {
        match (cur) {
            case some(n): { total = total + n.value; cur = n.next; }
            case null: { }
        }
    }
    printf("total %d\n", total);
    return 0;
}
