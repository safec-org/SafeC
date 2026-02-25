// concurrency.sc — M5: spawn + join with pthreads
extern int printf(const char *fmt, ...);

int main() {
    long long h0 = spawn || { printf("thread 0\n"); };
    long long h1 = spawn || { printf("thread 1\n"); };
    long long h2 = spawn || { printf("thread 2\n"); };

    join(h0);
    join(h1);
    join(h2);

    printf("all threads done\n");
    return 0;
}
