// A deferred free runs at return, so it doesn't stale the reference for
// the statements that follow it; a free that has already happened still
// does, and so does the deferred free itself when it runs on an
// already-freed pointer. Only the marked uses below are errors.
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
    defer dealloc(p);   // EXPECT-ERROR: after std::dealloc() freed it
    dealloc(p);
    dealloc(p); // EXPECT-ERROR: after std::dealloc() freed it
}

int main() { return deferred(); }
