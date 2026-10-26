#include <stdio.h>
int g_count = 3;
const int SIZE = 4;
double g_scale = 2.5;
int g_table[4] = {1, 2, 3, 4};
const char* g_msg = "global message";
struct Cfg { int w; int h; const char* title; };
struct Cfg g_cfg = {640, 480, "window"};
int next_id() {
    static int id = 100;
    id = id + 1;
    return id;
}
int main() {
    printf("%d %d %.1f %d\n", g_count, SIZE, g_scale, g_table[3]);
    unsafe { printf("%s\n", g_msg); }
    unsafe { printf("%d %d %s\n", g_cfg.w, g_cfg.h, g_cfg.title); }
    printf("%d %d %d\n", next_id(), next_id(), next_id());
    g_count = g_count * 2;
    printf("%d\n", g_count);
    int local[3] = {7, 8, 9};
    int m[2][3] = {{1, 2, 3}, {4, 5, 6}};
    printf("%d %d %d\n", local[2], m[1][2], m[0][1]);
    return 0;
}
