// arena.sc — M2: Arena runtime allocator
extern int printf(const char *fmt, ...);

region FramePool { capacity: 8192 }

struct Frame { double sample; int id; };

int main() {
    int *p1 = new<FramePool> int;
    unsafe { *p1 = 42; }
    unsafe { printf("p1 = %d\n", *p1); }

    int *p2 = new<FramePool> int;
    unsafe { *p2 = 100; }
    unsafe { printf("p2 = %d\n", *p2); }

    arena_reset<FramePool>();
    printf("arena reset OK\n");

    // Heap allocation
    int *h = new int;
    unsafe { *h = 7; }
    unsafe { printf("heap alloc = %d\n", *h); }

    return 0;
}
