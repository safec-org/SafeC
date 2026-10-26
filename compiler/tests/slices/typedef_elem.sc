// A typedef or newtype as a slice element type resolves like anywhere else:
// '[]Sample' elements are 'Sample'.
newtype Sample = int;
typedef int Word;
Sample first([]const Sample s) { Sample v = s[0]; return v; }
Word last([]Word w) { Word v = w[w.len - 1UL]; return v; }
// CHECK: define {{.*}}@first(
int main() {
    Sample a[2];
    a[0] = (Sample)3; a[1] = (Sample)4;
    Word b[2] = {5, 6};
    return (int)first(a[0..2]) + last(b[0..2]) - 9;
}
