struct S {
    static int n;   // EXPECT-ERROR: 'static' applies to methods only, not struct fields
};
int main() { return 0; }
