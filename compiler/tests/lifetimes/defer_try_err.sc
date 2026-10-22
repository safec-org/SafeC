// A plain defer also runs on a 'try' early return, after an explicit free.
void dealloc(&heap int p);
?int lookup_value(int k);

?int defer_after_free(int k) {
    &heap int p = new int;
    defer dealloc(p);                     // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
    dealloc(p);
    int v = try lookup_value(k);
    return v;
}

int main() { return 0; }
