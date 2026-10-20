// Raw pointers outside 'unsafe': every way of reaching memory through one,
// or of making one up, is an error. Each line below must produce exactly
// the error listed at the bottom.
struct P { int x; };

int  deref(int* q)                { return *q; }
void store(int* q)                { *q = 1; }
int  arrow(struct P* q)           { return q->x; }
int  subscript(int* q)            { return q[1]; }
int* arith(int* q)                { return q + 1; }
void step(int* q)                 { q++; }
void step_by(int* q)              { q += 2; }
long distance(int* q, int* r)     { return q - r; }
int* forge()                      { return (int*)4096; }
char* punned(int* q)              { return (char*)q; }
int* from_void(void* q)           { return (int*)q; }
&int to_ref(int* q)               { return (&int)q; }
int  take(&int r);
void to_ref_implicit(int* q)      { take(q); }

int main() { return 0; }

// EXPECT-ERROR: dereference of raw pointer requires 'unsafe' block
// EXPECT-ERROR: dereference of raw pointer requires 'unsafe' block
// EXPECT-ERROR: pointer member access requires 'unsafe' block
// EXPECT-ERROR: pointer subscript requires 'unsafe' block
// EXPECT-ERROR: raw pointer arithmetic requires 'unsafe' block
// EXPECT-ERROR: raw pointer arithmetic requires 'unsafe' block
// EXPECT-ERROR: raw pointer arithmetic requires 'unsafe' block
// EXPECT-ERROR: raw pointer arithmetic requires 'unsafe' block
// EXPECT-ERROR: cast from integer to raw pointer requires 'unsafe' block
// EXPECT-ERROR: cast between raw pointer types 'int*' and 'char*' requires 'unsafe' block
// EXPECT-ERROR: cast between raw pointer types 'void*' and 'int*' requires 'unsafe' block
// EXPECT-ERROR: cast from raw pointer to non-null reference '&int' requires 'unsafe' block
// EXPECT-ERROR: argument 1: cannot convert 'int*' to '&int'
