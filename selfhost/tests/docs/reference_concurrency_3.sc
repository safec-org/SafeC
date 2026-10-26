// extracted from safec-docs reference/concurrency.md (block 3)
extern int printf(const char* fmt, ...);
static ?&void g_ch = null;

?&int producer(?&int unused) {
    int value = 7;
    chan_send(g_ch, &value);
    chan_close(g_ch);
    return null;
}

int main() {
    g_ch = chan_create(4);
    long long h = spawn_scoped(producer, null);
    int received = 0;
    bool ok = chan_recv(g_ch, &received);
    join(h);
    printf("%d %d\n", (int)ok, received);
    return 0;
}
