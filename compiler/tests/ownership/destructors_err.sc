// drop() and 'consume' methods destroy *self: they must take apart every
// field of Drop type it owns (on some path: which fields hold a value can
// depend on run-time state).
struct Buf { int n; void drop(); };
void Buf::drop() { }

struct Owner { struct Buf a; struct Buf b; int state; void drop(); };
void Owner::drop() {                      // EXPECT-ERROR: 'Owner::drop()' never drops field 'b' (of Drop type 'Buf')
    if (self.state == 1) self.a.drop();
}

struct File2 { int fd; void drop(); int close() consume; int fd_of() const; };
void File2::drop() { self.close(); }
int File2::close() { int r = self.fd; self.fd = -1; return r; }
int File2::fd_of() const { return self.fd; }

int consume_then_use() {
    struct File2 f; f.fd = 3;
    int r = f.close();
    return r + f.fd_of();                 // EXPECT-ERROR: use of 'f' after it was dropped
}

// A non-'consume' method may not drop *self.
struct Bad { int n; void drop(); void reset(); };
void Bad::drop() { }
void Bad::reset() { self.drop(); }        // EXPECT-ERROR: cannot drop a value this function only borrows

int main() { return 0; }
