// Generic code is checked per instantiation: a T that is a Drop type is
// owned like any other value there.
struct Buf { int n; void drop(); };
void Buf::drop() { }
struct Buf mk(int n) { struct Buf b; b.n = n; return b; }

generic<T>
int ignore(T value) { return 1; }         // EXPECT-ERROR: value of Drop type 'Buf' leaks when the function returns: 'value' is never dropped

// drop(x) drops whatever T is (nothing for an int): the generic way.
generic<T>
int consume(T value) { drop(value); return 1; }

int main() {
    ignore(3);
    ignore(mk(1));
    consume(4);
    consume(mk(2));
    return 0;
}
