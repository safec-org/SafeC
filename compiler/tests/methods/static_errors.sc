// Misuses of static methods are rejected with a pointed message.
struct R {
    int x;
    static struct R make();
    int get() const;
};
struct R R::make() { return self; }   // EXPECT-ERROR: use of undeclared identifier 'self'
int R::get() const { return self.x; }
int main() {
    struct R r = R::make();
    struct R s = r.make();   // EXPECT-ERROR: 'make' is a static method of 'R'; call it as 'R::make(...)'
    int v = R::get();        // EXPECT-ERROR: 'R::get' is not a static method; call it on a value
    return 0;
}
