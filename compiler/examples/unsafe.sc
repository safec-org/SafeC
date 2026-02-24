// unsafe.sc — SafeC unsafe boundary model
//
// Demonstrates:
//   - unsafe { } blocks for raw pointer operations
//   - FFI calls with pointer arguments inside unsafe (see README §9.2)
//   - Explicit conversion from raw pointer to safe reference
//   - Pointer arithmetic only inside unsafe
//   - C-ABI interop
//
// extern signatures use raw C types (see README §9.1).
// Passing non-static pointers to C requires unsafe{} (see README §9.2).
// Receiving raw pointers from C happens inside unsafe{} (see README §9.6).

extern int   printf(char* fmt, ...);
extern void* malloc(long size);
extern void  free(void* ptr);

struct Header {
    int magic;
    int length;
}

// Takes a raw C pointer (from extern C API) and validates it in unsafe,
// then produces a safe reference for the rest of the function.
int readHeaderLength(void* raw) {
    int result = 0;
    unsafe {
        // Inside unsafe: raw pointer operations allowed
        Header* hdr = (Header*)raw;
        if (hdr != null) {
            result = hdr->length;
        }
    }
    return result;
}

// Demonstrate unsafe pointer arithmetic
void writeBytes(void* dst, int value, int count) {
    unsafe {
        char* p = (char*)dst;
        int i = 0;
        while (i < count) {
            // Pointer arithmetic: only inside unsafe
            p[i] = (char)value;
            i = i + 1;
        }
    }
}

// Raw allocation + safe wrapper pattern:
// Use unsafe to allocate, then return as &heap T
// (Full implementation would use explicit assert to tag the ref as &heap)
int* allocateInt(int value) {
    int* ptr = null;
    unsafe {
        // README §9.6: receiving raw pointer from C inside unsafe
        ptr = (int*)malloc((long)4);
        if (ptr != null) {
            ptr[0] = value;
        }
    }
    return ptr;
}

void freeInt(int* ptr) {
    unsafe {
        // README §9.2: passing pointer to C inside unsafe
        free((void*)ptr);
    }
}

int main() {
    // Allocate via unsafe, use result
    int* p = allocateInt(42);
    int val = 0;
    unsafe {
        if (p != null) val = p[0];
    }
    printf("Allocated value: %d\n", val);
    freeInt(p);

    return 0;
}
