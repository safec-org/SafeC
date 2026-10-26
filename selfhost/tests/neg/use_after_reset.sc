// use of an arena reference after arena_reset
// expect-error: use of 'p' (&arena<Pool> reference) after arena_reset<Pool>(), arena_destroy<Pool>(), or arena_free_to<Pool>() invalidated it
region Pool { capacity: 256 }
int main() {
    &arena<Pool> int p = new<Pool> int;
    *p = 7;
    arena_reset<Pool>();
    return *p;
}
