#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
int main() {
    char buf[64];
    int n = snprintf(&buf[0], 64UL, "x=%d y=%s", 5, "hi");
    unsafe { printf("%s (%d)\n", &buf[0], n); }
    char t[32] = "hello";
    unsafe {
        strcat(&t[0], " world");
        printf("%s %lu\n", &t[0], strlen(&t[0]));
        printf("%d %d\n", strcmp("abc", "abd") < 0, strncmp("abcdef", "abcxyz", 3UL));
        char* p = strchr(&t[0], 'w');
        if (p != null) { printf("%s\n", p); }
        char* d = strdup("dup");
        printf("%s\n", d);
        free(d);
    }
    printf("%d %d %d\n", atoi("123"), atoi("-45"), (int)strtol("ff", null, 16));
    printf("%d %d %c %c\n", isdigit('7'), isalpha('7'), (char)toupper('a'), (char)tolower('Q'));
    const char* names[3];
    names[0] = "zero"; names[1] = "one"; names[2] = "two";
    for (int i = 0; i < 3; i = i + 1) { printf("%s ", names[i]); }
    printf("\n");
    printf("%5s|%-5s|%.2s\n", "ab", "cd", "efgh");
    return 0;
}
