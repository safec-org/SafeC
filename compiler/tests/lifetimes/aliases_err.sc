// Frees reach every alias of the allocation: copies of copies, copies made
// on either side of a branch, struct fields (directly or through a copied
// struct) and array elements (an unknown index may be any element).
void dealloc(&heap int p);
struct Box { &heap int v; int n; };

int multi_hop() {
    &heap int p = new int;
    &heap int q = p;
    &heap int r = q;
    &heap int s = r;
    dealloc(s);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

int joined_alias(int k) {
    &heap int p = new int;
    &heap int q = new int;
    &heap int r = p;
    if (k) r = q;
    dealloc(r);
    return *p + *q;                       // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}
// (one directive per error on the line)
// EXPECT-ERROR: use of 'q' (&heap reference) after std::dealloc() freed it

int field_alias() {
    struct Box b;
    b.v = new int;
    &heap int alias = b.v;
    dealloc(alias);
    return *b.v;                          // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

int copied_struct() {
    struct Box b;
    b.v = new int;
    struct Box c = b;
    dealloc(c.v);
    return *b.v;                          // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

int unknown_index_read(int i) {
    &heap int arr[4];
    arr[0] = new int;
    arr[1] = new int;
    dealloc(arr[1]);
    return *arr[0] + *arr[i];             // EXPECT-ERROR: use of a '&heap T' value read through a struct field/array element after std::dealloc() freed it
}

int unknown_index_store(int i) {
    &heap int arr[4];
    &heap int p = new int;
    arr[i] = p;
    dealloc(arr[2]);
    return *p;                            // EXPECT-ERROR: use of 'p' (&heap reference) after std::dealloc() freed it
}

int double_free_through_alias() {
    &heap int p = new int;
    &heap int q = p;
    dealloc(p);
    dealloc(q);                           // EXPECT-ERROR: use of 'q' (&heap reference) after std::dealloc() freed it
    return 0;
}

int main() { return 0; }
