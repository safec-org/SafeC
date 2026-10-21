// An out-of-line method definition must match its declaration in the
// struct body (it used to be accepted whatever the parameter types).
struct Dev {
    void* ctx;
    static struct Dev new(&long ctx);
    int get(&long y) const;
    int k() const;
    int two(int a, int b);
};
struct Dev Dev::new(&int ctx) { struct Dev d; unsafe { d.ctx = (void*)ctx; } return d; }   // EXPECT-ERROR: definition of 'Dev::new' doesn't match its declaration in the struct body: parameter 1 is declared '&long', defined '&int'
int Dev::get(&int y) const { return *y; }   // EXPECT-ERROR: definition of 'Dev::get' doesn't match its declaration in the struct body: parameter 1 is declared '&long', defined '&int'
int Dev::k() { return 1; }                  // EXPECT-ERROR: definition of 'Dev::k' doesn't match its declaration in the struct body: it is declared const
int Dev::two(int a) { return a; }           // EXPECT-ERROR: definition of 'Dev::two' doesn't match its declaration in the struct body: it is declared with 2 parameter(s)
int main() { return 0; }
