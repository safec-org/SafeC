// The IR names its target triple for a host build too (it used to only
// for --target): clang linking it (--emit-bin) otherwise warned
// "overriding the module target triple" on every build.
// CHECK: target datalayout = "{{.+}}"
// CHECK: target triple = "{{.+}}"
int main() { return 0; }
