// A semantic error inside an #include'd header is reported at the header's
// path and line, not the including file's.
#include "inc/bad_decl.h"
// EXPECT-ERROR-IN: inc/bad_decl.h:4: undeclared_in_header

int main() { return header_fn(); }
