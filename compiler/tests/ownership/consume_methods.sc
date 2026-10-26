// A 'consume' method (and drop()) destroys its receiver; defer runs the
// drop at every return.
// (Defined elsewhere, so the calls stay visible at -O2.)
struct File3 { int fd; void drop(); int close() consume; int fd_of() const; };
struct File3 open3(int fd);

// CHECK-LABEL: define {{.*}}@closes(
// CHECK: call {{.*}}@File3_close(
int closes(int fd) {
    struct File3 f = open3(fd);
    int n = f.fd_of();
    return n + f.close();
}

// CHECK-LABEL: define {{.*}}@deferred(
// CHECK: call void @File3_drop(
int deferred(int fd, int c) {
    struct File3 f = open3(fd);
    defer f.drop();
    if (c) return 1;
    return f.fd_of();
}

int main() { return closes(3) + deferred(4, 0); }
