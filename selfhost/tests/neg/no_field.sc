// unknown struct field
// expect-error: struct 'P' has no field 'z'
struct P { int x; int y; };
int main() { struct P p; p.x = 1; p.y = 2; return p.z; }
