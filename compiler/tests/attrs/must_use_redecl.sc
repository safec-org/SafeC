// must_use / warn_unused_result on any declaration of a function applies to
// it: a prototype's attribute was lost when the definition that followed
// (or preceded) it didn't repeat it.
must_use int f(int x);
int g(int x) __attribute__((warn_unused_result));
int f(int x) { return x; }
int g(int x) { return x; }
int h(int x) { return x; }
must_use int h(int x);
int plain(int x) { return x; }

// CHECK-LABEL: define {{.*}}i32 @main(
int main() {
    f(1);          // EXPECT-WARNING: return value of 'f' should not be ignored
    g(2);          // EXPECT-WARNING: return value of 'g' should not be ignored
    h(3);          // EXPECT-WARNING: return value of 'h' should not be ignored
    plain(4);
    return f(0) + g(0) + h(0);
}
