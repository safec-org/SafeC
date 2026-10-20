// A deferred free runs at scope exit, so it doesn't stale the reference
// for the statements that follow it; a free that has already happened
// still does. Only the two marked uses below are errors.
void dealloc(&heap int p);

int deferred() {
    &heap int p = new int;
    defer dealloc(p);
    *p = 42;
    return *p;
}

int freed_then_used() {
    &heap int p = new int;
    dealloc(p);
    return *p;  // EXPECT-ERROR: after std::dealloc() freed it
}

void double_free() {
    &heap int p = new int;
    defer dealloc(p);
    dealloc(p);
    dealloc(p); // EXPECT-ERROR: after std::dealloc() freed it
}

int main() { return deferred(); }
