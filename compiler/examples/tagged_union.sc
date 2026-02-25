extern int printf(const char *fmt, ...);

union Shape {
    double radius;
    double side;
};

double area(Shape s) {
    match (s) {
        case .radius(r): return 3.14159 * r * r;
        case .side(sd):  return sd * sd;
        default:         return 0.0;
    }
}

int main() {
    Shape c = Shape.radius(5.0);
    Shape sq = Shape.side(3.0);
    printf("circle: %f\n", area(c));
    printf("square: %f\n", area(sq));
    return 0;
}
