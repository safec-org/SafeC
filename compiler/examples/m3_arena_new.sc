// M3: Arena allocation with new<R> and arena_reset<R>
extern int printf(const char *fmt, ...);

region AudioPool { capacity: 4096 }

struct AudioBuffer {
    int sample_rate;
    int num_channels;
};

int main() {
    // Allocate from arena
    int *p = new<AudioPool> int;
    printf("Arena new<int>: p = %p\n", p);
    
    // Reset the arena
    arena_reset<AudioPool>();
    printf("Arena reset OK\n");
    
    // Heap allocation
    int *h = new int;
    printf("Heap new<int>: h = %p\n", h);
    
    return 0;
}
