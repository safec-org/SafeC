// an integer literal that doesn't fit
// expect-error: type mismatch in variable declaration: cannot convert 'int' to 'uint8'
int main() { unsigned char c = 300; return (int)c; }
