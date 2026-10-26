// extracted from safec-docs reference/namespaces.md (block 1)
namespace std {
    int square(int x) { return x * x; }
    int counter = 41;

    namespace inner {
        int triple(int x) { return x * 3; }
    }
}

int main(void) {
    int a = std::square(6);          // qualified
    int b = std::inner::triple(4);   // nested namespace, qualified
    std::counter = std::counter + 1;

    int c = square(6);               // unqualified fallback also resolves
    return 0;
}
