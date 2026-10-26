// What spawn / spawn_scoped / chan_send let a thread see.
struct Counter { int n; };
static struct Counter g_c;
static int g_ro = 5;
?&Counter work(?&Counter c) { match (c) { case some(p): p.n = p.n + 1; case null: {} } return null; }
?&int read_only(?&int c) { return null; }

void detached_local() {
    struct Counter c; c.n = 0;
    long long h = spawn(work, &c);        // EXPECT-ERROR: spawn: the argument '&stack Counter' refers to data the thread may outlive
    join(h);
}

void static_unsynchronized() {
    long long h = spawn(work, &g_c);      // EXPECT-ERROR: but it is not safe to write concurrently
    join(h);
}

void static_read_only() {
    long long h = spawn(read_only, (const &static int)&g_ro);
    join(h);
}

void scoped_ok() {
    struct Counter c; c.n = 0;
    long long h = spawn_scoped(work, &c);
    join(h);
    c.n = c.n + 1;
}

void scoped_touch() {
    struct Counter c; c.n = 0;
    long long h = spawn_scoped(work, &c);
    c.n = 5;                              // EXPECT-ERROR: 'c' is lent to the thread started here by spawn_scoped
    join(h);
}

void scoped_scope_end() {
    long long h = 0;
    {
        struct Counter c; c.n = 0;
        h = spawn_scoped(work, &c);
    }                                     // EXPECT-ERROR: 'c' goes away while the thread started by spawn_scoped may still be using it
    join(h);
}

// Returning joins every scoped thread first: a function-level local may
// stay lent until then.
void scoped_until_return() {
    struct Counter c; c.n = 0;
    long long h = spawn_scoped(work, &c);
}

struct Holds { int *p; };
void send_raw(?&void ch) {
    struct Holds h; h.p = (int*)0;
    chan_send(ch, &h);                    // EXPECT-ERROR: chan_send: a value of type 'Holds' can't be sent to another thread
}

int main() { return 0; }
