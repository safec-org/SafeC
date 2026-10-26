// Drop forms: drop() and consume methods, drop(x) glue, needs_drop(T), conditional
// Drop (Result), forget(x), std::replace.
#include <std/collections/vec.h>
#include <std/collections/string.h>
#include <std/result.h>
#include <std/io.h>
extern int printf(const char* fmt, ...);
struct Buf { int n; void drop(); };
void Buf::drop() { printf("drop %d\n", self.n); }
struct Fd { int fd; void drop(); int close() consume; };
int Fd::close() consume { int r = self.fd; self.fd = -1; return r; }
void Fd::drop() { self.close(); }
struct Pair { struct Buf a; int x; struct Buf b; };
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }
generic<T> int owns(T v) { int r = needs_drop(T); drop(v); return r; }
int main() {
    struct Vec<struct String> v = Vec::new();
    v.push(String::from("a"));
    v.push(String::from("b"));
    struct String old = std::replace(v.at(0UL), String::from("c"));
    printf("%s %s\n", old.as_ptr(), v.at(0UL).as_ptr());
    old.drop();
    v.drop();
    struct Pair p; p.a = mk(1); p.x = 0; p.b = mk(2);
    drop(p);
    struct Buf arr[2]; arr[0] = mk(3); arr[1] = mk(4);
    drop(arr);
    printf("%d %d\n", owns(mk(5)), owns(7));
    struct Result<int, int> r = Result::new_ok(3);
    printf("%d %d\n", needs_drop(struct Result<int, int>), needs_drop(struct Result<struct Buf, int>));
    printf("%d\n", r.unwrap());
    struct Buf f = mk(9);
    unsafe { forget(f); }
    struct Fd d; d.fd = 4;
    printf("closed %d\n", d.close());
    return 0;
}
