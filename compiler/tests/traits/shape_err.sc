// A trait's declaration fixes each method's shape: a type whose method has
// a different parameter count or static-ness doesn't satisfy it, and the
// error says why (through supertraits too). A failed bound stops there,
// without repeating the missing methods from inside the generic body.
trait Proc { double process(double x); }
trait Reset { void reset(); }
trait Filt : Proc + Reset {}
trait Bad { Sample process(Sample x); }   // EXPECT-ERROR: unknown type 'Sample' in trait 'Bad' method 'process'

struct Gen { double v; double process(); void reset(); };
double Gen::process() { return self.v; }
void Gen::reset() { self.v = 0.0; }

generic<F: Filt> struct Holder { F f; double run(double x); };
generic<F: Filt> double Holder::run(double x) { return self.f.process(x); }
generic<F: Filt> double once(&F f, double x) { f.reset(); return f.process(x); }

int main() {
    struct Gen g; g.v = 2.0;
    double s = once(&g, 2.0);   // EXPECT-ERROR: type 'Gen' does not satisfy constraint 'Filt' for generic parameter 'F' ('process' takes 0 parameter(s), the trait 1, from 'Proc')
    struct Holder<struct Gen> h;   // EXPECT-ERROR: type 'Gen' does not satisfy constraint 'Filt' for generic parameter 'F' of struct 'Holder' ('process' takes 0 parameter(s), the trait 1, from 'Proc')
    return 0;
}
