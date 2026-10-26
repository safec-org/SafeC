// A field can reference a struct that is defined further down the file
// ('?&struct Mount m;' before 'struct Mount'): its members are reachable
// through the field once the definition is seen.
struct Node {
    int id;
    ?&struct Mount mount;
};
struct Mount {
    int  tag;
    char name[8];
    int  depth;
};
int mount_depth(&Node n) {
    match (n.mount) {
        case some(m): return m.depth + m.tag;
        case null: return -1;
    }
    return -1;
}
// CHECK: define {{.*}}@mount_depth(
// CHECK: getelementptr inbounds nuw i8, ptr %{{.*}}, i64 12
int main() {
    struct Mount m; m.tag = 1; m.depth = 2;
    struct Node n; n.id = 0; n.mount = &m;
    return mount_depth(&n) - 3;
}
