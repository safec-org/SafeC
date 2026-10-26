#include <stdio.h>
enum Status : int { OK = 200, NotFound = 404, ServerError = 500 }
enum Day { Monday, Tuesday, Wednesday, Thursday, Friday }
union Shape { double radius; double side; }
double area(union Shape s) {
    return match (s) {
        case .radius(r): 3.14159 * r * r,
        case .side(sd):  sd * sd,
    };
}
int severity_of(enum Status s) {
    return match (s) {
        case OK:       0,
        case NotFound: 1,
        default:       2,
    };
}
void classify(int status_code) {
    match (status_code) {
        case 200:      printf("OK\n");
        case 400..499: printf("client error\n");
        case 500..599: printf("server error\n");
        default:       printf("other\n");
    }
}
void day_kind(int day) {
    match (day) {
        case 1, 7:          printf("weekend\n");
        case 2, 3, 4, 5, 6: printf("weekday\n");
        default:            printf("?\n");
    }
}
void ch(char c) {
    match (c) {
        case 'a'...'z': printf("lowercase\n");
        case 'A'..'Z':  printf("uppercase\n");
        case '0'...'9': printf("digit\n");
        default:        printf("other\n");
    }
}
int main() {
    enum Status s = NotFound;
    printf("%d\n", (int)s);
    union Shape circle = Shape.radius(2.0);
    union Shape square = Shape.side(3.0);
    printf("%.5f %.1f\n", area(circle), area(square));
    printf("%d %d %d\n", severity_of(OK), severity_of(NotFound), severity_of(ServerError));
    classify(200); classify(404); classify(503); classify(42);
    day_kind(1); day_kind(3); day_kind(9);
    ch('q'); ch('Q'); ch('5'); ch('#');
    enum Day d = Wednesday;
    match (d) {
        case Monday: printf("start\n");
        case Wednesday: printf("mid\n");
        default: printf("later\n");
    }
    return 0;
}
