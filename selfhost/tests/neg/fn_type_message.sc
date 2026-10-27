// a function reference prints as the 'fn R(Args)' that spells it
// expect-error: cannot convert 'fn int(int)' to 'int'
int sq(int x) { return x * x; }
int main() {
    fn int(int) f = sq;
    int y = f;
    return y;
}
