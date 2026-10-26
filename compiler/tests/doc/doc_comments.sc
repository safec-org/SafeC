// SAFEC-DOC-JSON
//! A test module for --emit-doc-json: the file's own doc.

// ── Section rule: not part of any doc ────────────────────────────────────

/// A point in the plane.
///
/// Markdown, with [Point::norm2] as an intra-doc link.
struct Point {
    int x;   // horizontal
             // (continued on the next line)
    /// vertical
    int y;

    // squared length
    int norm2() const;
    static struct Point origin();
};

/** Colors, documented with a block comment. */
enum Color { RED = 1, GREEN, /// the second one
             BLUE };

// Plain-comment (std convention) documentation of a generic function.
generic<T: Numeric>
T twice(T v) { return v + v; }

// Out-of-line definitions lend their doc to the declaration.
int Point::norm2() const { return self.x * self.x + self.y * self.y; }

#define POINT_MAX 100   // the largest coordinate

/// @hidden
int secret_helper() { return 0; }

static int file_local() { return 1; }

namespace geo {
    // A namespaced function.
    int area(int w, int h) { return w * h; }
}

trait Shape {
    // The shape's area.
    int area() const;
}

// CHECK: "path":"doc_comments.sc","doc":"A test module for --emit-doc-json: the file's own doc.","md":true
// CHECK: "kind":"struct","name":"Point","file":"doc_comments.sc","line":9,"sig":"struct Point","doc":"A point in the plane.\n\nMarkdown, with [Point::norm2] as an intra-doc link.","md":true
// CHECK: "name":"x","line":10,"sig":"int x","type":"int","doc":"horizontal\n(continued on the next line)"
// CHECK: "name":"y","line":13,"sig":"int y","type":"int","doc":"vertical","md":true
// CHECK: "name":"norm2","line":16,"sig":"int norm2() const"{{.*}}"doc":"squared length","def_file":"doc_comments.sc","def_line":29
// CHECK: "name":"origin","line":17,"sig":"static struct Point origin()"
// CHECK: "kind":"enum","name":"Color"{{.*}}"doc":"Colors, documented with a block comment.","md":true
// CHECK: "name":"RED","line":21,"value":1}
// CHECK: "name":"GREEN","line":21,"value":2,"doc":"the second one","md":true
// CHECK: "kind":"function","name":"twice","file":"doc_comments.sc","line":26,"sig":"generic<T: Numeric> T twice(T v)","doc":"Plain-comment (std convention) documentation of a generic function."{{.*}}"generics":[{"name":"T","bounds":["Numeric"]}],"params":[{"name":"v","type":"T"}],"ret":"T"
// CHECK: "kind":"macro","name":"POINT_MAX"{{.*}}"sig":"#define POINT_MAX 100","doc":"the largest coordinate"
// CHECK: "kind":"function","name":"secret_helper"{{.*}}"hidden":true
// CHECK-NOT: file_local
// CHECK: "kind":"function","name":"area","ns":"geo"{{.*}}"doc":"A namespaced function."
// CHECK: "kind":"trait","name":"Shape"
// CHECK: "name":"area","line":45,"sig":"int area() const"{{.*}}"doc":"The shape's area."
