// SAFEC-ARGS: -I %root/std -I %root --compat-preprocessor --no-incremental
// MAX-SECONDS: 3
// A run of insert() calls on one tree of must-drop values: the memory-flow
// check applies insert()'s summary once per call, and every call can see
// every node made so far. Its state used to nest a node inside a node's
// 'val' once per call (paths through the summary's collapsed entry paths)
// and to multiply the caller cells per dereference, so the time doubled
// with each call (7 calls took 46 s). It must stay well under a second.
#include <std/collections/bst.h>
#include <std/cmp.h>

struct Tracked { int id; void drop(); };
void Tracked::drop() { }
struct Tracked tracked(int id) { struct Tracked t; t.id = id; return t; }

// CHECK: define {{.*}}i32 @main(
int main() {
    struct BST<int, struct Tracked, Less<int>> bt = BST::new();
    bt.insert(1, tracked(1));
    bt.insert(2, tracked(2));
    bt.insert(3, tracked(3));
    bt.insert(4, tracked(4));
    bt.insert(5, tracked(5));
    bt.insert(6, tracked(6));
    bt.insert(7, tracked(7));
    bt.insert(8, tracked(8));
    bt.insert(9, tracked(9));
    bt.insert(10, tracked(10));
    bt.insert(11, tracked(11));
    bt.insert(12, tracked(12));
    bt.insert(13, tracked(13));
    bt.insert(14, tracked(14));
    bt.insert(15, tracked(15));
    bt.insert(16, tracked(16));
    bt.insert(17, tracked(17));
    bt.insert(18, tracked(18));
    bt.insert(19, tracked(19));
    bt.insert(20, tracked(20));
    bt.insert(21, tracked(21));
    bt.insert(22, tracked(22));
    bt.insert(23, tracked(23));
    bt.insert(24, tracked(24));
    bt.drop();
    return 0;
}
