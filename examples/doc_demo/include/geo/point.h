#pragma once
//! Points and vectors in the plane.
//!
//! ```c
//! struct Point p = Point::new(3, 4);
//! long long d = p.norm2();   // 25
//! ```
#include <std/collections/string.h>

namespace geo {

/// A point with integer coordinates.
///
/// See [Point::norm2] for its squared length, and [`std::String`] for
/// [Point::describe]'s result.
struct Point {
    int x;   ///< horizontal coordinate
    int y;   ///< vertical coordinate

    /// A point at (`x`, `y`).
    static struct Point new(int x, int y);
    /// The squared distance from the origin.
    long long norm2() const;
    /// "(x, y)" as a [String].
    struct String describe() const;
};

/// Kinds of shape [shape_area] knows.
enum ShapeKind { SHAPE_SQUARE = 1, SHAPE_CIRCLE };

/// The largest coordinate a [Point] may hold.
#define GEO_MAX_COORD 1000000

/// Area of a shape of `kind` and size `s`, rounded down.
///
/// # Examples
///
/// ```c
/// long long a = geo::shape_area(SHAPE_SQUARE, 3);   // 9
/// ```
long long shape_area(enum ShapeKind kind, int s);

/// Something with an area.
trait HasArea {
    /// The area.
    long long area() const;
}

}
