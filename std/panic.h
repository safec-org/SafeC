// SafeC Standard Library — Optional Panic System
// Opt-in; does nothing unless panic_set_handler is called.
// In freestanding mode defaults to infinite loop; hosted mode calls abort().
#pragma once

// Panic handler function type: fn(message, file, line) → noreturn
typedef void (*PanicHandler)(const char* msg, const char* file, int line);

// Install a custom panic handler. Pass NULL to reset to default.
void panic_set_handler(PanicHandler handler);

// Trigger a panic with message, file, line.
// Never returns.
void panic_at(const char* msg, const char* file, int line);

// Convenience macro.
#define PANIC(msg)  panic_at((msg), __FILE__, __LINE__)

// Assert that condition is true; panics with expression string if false.
#define PANIC_ASSERT(cond)  do { if (!(cond)) { PANIC(#cond); } } while(0)
