// Diagnostics after '#line N "file"' are reported at the presumed file and
// line.
#line 40 "generated.sc"
int main() { return not_declared_here; }
// EXPECT-ERROR-IN: generated.sc:40: use of undeclared identifier 'not_declared_here'
