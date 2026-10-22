// C's simple escape sequences: '\f' and '\v' (and '\a', '\b', '\e') used to
// lex as the plain letters 'f', 'v', ... in char literals and as a
// backslash plus the letter in strings, so e.g. std's isspace-style
// checks treated 'f' and 'v' as whitespace.
// CHECK-LABEL: define {{.*}}@form_feed(
// CHECK: ret i32 12
int form_feed() { return (int)'\f'; }
// CHECK-LABEL: define {{.*}}@vtab(
// CHECK: ret i32 11
int vtab() { return (int)'\v'; }
// CHECK-LABEL: define {{.*}}@bell(
// CHECK: ret i32 7
int bell() { return (int)'\a'; }
// CHECK-LABEL: define {{.*}}@backspace(
// CHECK: ret i32 8
int backspace() { return (int)'\b'; }
// CHECK-LABEL: define {{.*}}@escape(
// CHECK: ret i32 27
int escape() { return (int)'\e'; }
// CHECK-LABEL: define {{.*}}@dquote(
// CHECK: ret i32 34
int dquote() { return (int)'\"'; }

// Strings hold the control bytes, not backslash pairs.
// CHECK-LABEL: define {{.*}}@string_escapes(
// CHECK: ret i32 41
int string_escapes() {
    const char* s = "\f\v\a\b";   // 12 + 11 + 7 + 8 = 38, 4 bytes
    int sum = 0;
    int n = 0;
    unsafe {
        while (s[n] != '\0') { sum = sum + (int)s[n]; n = n + 1; }
    }
    return sum + n - 1;
}
