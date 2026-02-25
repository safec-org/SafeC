// M5: Trait enforcement via generic constraints
extern int printf(const char *fmt, ...);

struct Counter {
    int value;
    Counter operator+(Counter other) const;
    Counter operator-(Counter other) const;
};

Counter Counter::operator+(Counter other) const {
    Counter result;
    result.value = self.value + other.value;
    return result;
}

Counter Counter::operator-(Counter other) const {
    Counter result;
    result.value = self.value - other.value;
    return result;
}

// Generic function that requires Add trait (operator+)
generic<T>
T add_two(T a, T b) {
    return a + b;
}

int main() {
    // Test with int (builtin Numeric)
    int x = add_two(3, 4);
    printf("add_two(3, 4) = %d\n", x);
    
    // Test with Counter struct (has operator+)
    Counter a;
    a.value = 10;
    Counter b;
    b.value = 5;
    Counter c = add_two(a, b);
    printf("add_two(Counter(10), Counter(5)).value = %d\n", c.value);
    
    return 0;
}
