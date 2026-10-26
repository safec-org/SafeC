// assignment to a const variable
// expect-error: cannot assign to const variable 'k'
int main() { const int k = 1; k = 2; return k; }
