// extracted from safec-docs reference/functional.md (block 1)
extern int printf(const char* fmt, ...);
generic<T>
void map_inplace(T* arr, int n, fn T(T) f) {
    int i = 0;
    while (i < n) {
        unsafe { arr[i] = f(arr[i]); }
        i = i + 1;
    }
}

generic<T>
T fold(T* arr, int n, T init, fn T(T, T) f) {
    T acc = init;
    int i = 0;
    while (i < n) {
        T v;
        unsafe { v = arr[i]; }
        acc = f(acc, v);
        i = i + 1;
    }
    return acc;
}

pure int square(int x) { return x * x; }
pure int add(int a, int b) { return a + b; }

int main() {
    int nums[5] = {1, 2, 3, 4, 5};
    int* p;
    unsafe { p = (int*)&nums[0]; }

    map_inplace(p, 5, square);              // nums -> 1 4 9 16 25
    int total = fold(p, 5, 0, add);          // 55
    printf("sum of squares = %d\n", total);
    return 0;
}
