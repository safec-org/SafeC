// concurrency.sc — spawn + join with pthreads (C-style function pointers)
extern int printf(const char *fmt, ...);

void* worker0(void* arg) { printf("thread 0\n"); return (void*)0; }
void* worker1(void* arg) { printf("thread 1\n"); return (void*)0; }
void* worker2(void* arg) { printf("thread 2\n"); return (void*)0; }

int main() {
    long long h0 = spawn(worker0, 0);
    long long h1 = spawn(worker1, 0);
    long long h2 = spawn(worker2, 0);

    join(h0);
    join(h1);
    join(h2);

    printf("all threads done\n");
    return 0;
}
