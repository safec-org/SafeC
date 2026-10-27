// Diagnostics from inside an #include'd header carry that header's path and
// line. (The preprocessor recorded them with a pointer into a string freed
// when the #include returned, so stored diagnostics, e.g. the language
// server's, named garbage or the wrong file.)
#include "inc/bad_header.h"
// EXPECT-ERROR-IN: inc/bad_header.h:3: #error raised inside a header
// EXPECT-ERROR-IN: inc/bad_header.h:4: cannot find include file 'no_such_header_anywhere.h'

int main() { return 0; }
