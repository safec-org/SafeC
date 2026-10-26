// goto: backward and forward jumps, out of nested loops, into a loop body
// (a resumable state machine), into a switch case, and labels after returns
#include <stdio.h>

int count_down(int n) {
    int steps = 0;
again:
    if (n > 0) { n = n - 1; steps = steps + 1; goto again; }
    return steps;
}

int find_pair(int target) {
    int found = -1;
    for (int i = 0; i < 10; i = i + 1) {
        for (int j = 0; j < 10; j = j + 1) {
            if (i * j == target) { found = i * 100 + j; goto done; }
        }
    }
done:
    return found;
}

struct Gen { int state; int i; int acc; };

// yields 0, 1, 4, 9, ... until i reaches the limit; resumes inside the loop
int next_square(&struct Gen g, int limit) {
    switch (g.state) {
    case 0: break;
    case 1: goto resume;
    default: return -1;
    }
    g.i = 0;
    while (g.i < limit) {
        g.acc = g.i * g.i;
        g.state = 1;
        return g.acc;
resume:
        g.i = g.i + 1;
    }
    g.state = 2;
    return -1;
}

int into_case(int k) {
    int r = 0;
    if (k > 5) goto big;
    switch (k) {
    case 1: r = 10; break;
    case 2:
big:
        r = r + 20;
        break;
    default: r = 30;
    }
    return r;
}

int main() {
    printf("%d\n", count_down(7));
    printf("%d %d\n", find_pair(42), find_pair(97));
    struct Gen g;
    g.state = 0; g.i = 0; g.acc = 0;
    int v = next_square(&g, 5);
    while (v >= 0) { printf("%d ", v); v = next_square(&g, 5); }
    printf("| state %d\n", g.state);
    printf("%d %d %d %d\n", into_case(1), into_case(2), into_case(3), into_case(9));
    return 0;
}
