// outliving_refs.sc — SafeC outliving (region-less) references, '?&T'
//
// Demonstrates:
//   - '?&T': a nullable reference with no region qualifier, for pointers
//     that cross an 'extern' (C-ABI) function boundary and may be
//     retained by the C side past the current call — see README's
//     "Outliving references (?&T)" section for the full rationale.
//   - Implicit, unsafe-free conversion both ways: &region T -> ?&T,
//     raw T* -> ?&T, ?&T -> T*.
//   - Reading a '?&T' out via 'match' (safe) and '!' (unsafe-gated),
//     same grammar as any other nullable reference.
//
// extern signatures use raw C types (see README §9.1).

extern int printf(char* fmt, ...);

struct Widget {
    int value;
}

// Stands in for a real 'extern' C API that registers a callback/handle and
// conceptually retains the pointer past this call returning (defined here,
// not 'extern', purely so this example links standalone).
int c_register_callback(struct Widget* w) {
    int v;
    unsafe { v = w->value; }
    return v;
}

// A parameter typed '?&Widget': no region to declare, no unsafe needed to
// receive or hold it.
int useOutliving(?&Widget w) {
    match (w) {
        case null: return -1;
        case some(v): return v.value; // 'v' is a plain 'Widget' value — no unsafe
    }
}

int main() {
    Widget local;
    local.value = 42;

    // &stack Widget -> ?&Widget: implicit, no unsafe, no region check —
    // '?&T' doesn't track a region, so it accepts any region's reference.
    ?&Widget wref = &local;
    printf("stack -> outliving: %d\n", useOutliving(wref));

    // null -> ?&Widget: implicit, the empty case.
    ?&Widget wnull = null;
    printf("null case: %d\n", useOutliving(wnull));

    // ?&Widget -> struct Widget* (as a call argument to a C-ABI-style
    // function): implicit, no unsafe.
    int registered = c_register_callback(wref);
    printf("registered: %d\n", registered);

    // '!' force-unwrap still requires 'unsafe', and yields a plain Widget
    // lvalue (not a reference) — same as any other nullable reference.
    int direct;
    unsafe { direct = wref!.value; }
    printf("force-unwrap: %d\n", direct);

    return 0;
}
