// channels.sc — typed channel communication + scoped spawn demo
// Compile: ./build/safec examples/channels.sc --emit-llvm -o /dev/null

// Channel API (backed by __safec_chan_* runtime functions):
//   chan_create(capacity) → void*     Create a bounded channel
//   chan_send(ch, val_ptr) → bool     Send value (blocks if full)
//   chan_recv(ch, out_ptr) → bool     Receive value (returns false if closed)
//   chan_close(ch) → void             Close the channel

// Producer: writes integers into a channel
void* producer(void* arg) {
    void* ch = arg;
    int val = 42;
    unsafe { chan_send(ch, (void*)&val); }
    int val2 = 100;
    unsafe { chan_send(ch, (void*)&val2); }
    return (void*)0;
}

// Consumer: reads integers from a channel
void* consumer(void* arg) {
    void* ch = arg;
    int received = 0;
    unsafe { chan_recv(ch, (void*)&received); }
    return (void*)0;
}

// Demonstrate scoped spawn: thread is guaranteed to join before scope exit
void scoped_demo(void* ch) {
    // spawn_scoped guarantees the thread joins before this scope exits
    long long handle = spawn_scoped(producer, ch);

    // Main thread can receive
    int result = 0;
    unsafe { chan_recv(ch, (void*)&result); }

    // Thread auto-joins when scope exits (via deferred join)
}

int main() {
    void* ch = chan_create(16);
    scoped_demo(ch);
    chan_close(ch);
    return 0;
}
