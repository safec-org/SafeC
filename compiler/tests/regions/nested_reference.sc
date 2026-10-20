// References to references: '&&T' (the lexer's single '&&' token), '& &T',
// '&?&T' and '?&?&T' all parse as a reference whose referent is itself a
// reference, and writing through the outer one stores the inner pointer.
static int g = 7;

void set_always(&&int p)  { *p = &g; }
void set_spaced(& &int p) { *p = &g; }
void set_maybe(&?&int p)  { *p = null; }
int  opt_present(?&?&int p) { return p.is_null() ? 0 : 1; }

// CHECK-LABEL: define {{.*}}@set_always(
// CHECK: store ptr @g
// CHECK-LABEL: define {{.*}}@set_spaced(
// CHECK: store ptr @g

int main() {
    int x = 1;
    &int r = &x;
    set_always(&r);
    set_spaced(&r);
    int y = 2;
    ?&int m = &y;
    set_maybe(&m);
    return opt_present(null);
}
