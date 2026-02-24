// preproc.sc — SafeC preprocessor demonstration
//
// Shows:
//   - Object-like #define constants (safe mode)
//   - #ifdef / #ifndef / #if / #else / #elif / #endif
//   - #undef
//   - __FILE__ / __LINE__
//   - #error / #warning
//
// extern signatures use raw C types (see README §9.1).

extern int printf(char* fmt, ...);

// ── Object-like macros ────────────────────────────────────────────────────────
// Allowed: expand to constant expressions only
#define BUFFER_SIZE   256
#define MAX_CHANNELS  8
#define VERSION_MAJOR 1
#define VERSION_MINOR 0
#define VERSION       ((VERSION_MAJOR * 100) + VERSION_MINOR)

// SafeC encourages const instead of #define for new code:
//   const int BUFFER_SIZE = 256;
// But #define is supported for C interoperability.

// ── Platform / feature flags ──────────────────────────────────────────────────
#define SAFEC_PLATFORM_MACOS 1

#ifdef SAFEC_PLATFORM_MACOS
#define PLATFORM_NAME "macOS"
#else
#define PLATFORM_NAME "unknown"
#endif

// ── Arithmetic in #if ─────────────────────────────────────────────────────────
#if VERSION >= 100
#define FEATURE_ENABLED 1
#else
#define FEATURE_ENABLED 0
#endif

// ── #ifndef guard pattern ─────────────────────────────────────────────────────
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

// ── #undef ────────────────────────────────────────────────────────────────────
#define TEMP_MACRO 42
#undef  TEMP_MACRO
// After undef, TEMP_MACRO is gone.

// ── #elif chain ───────────────────────────────────────────────────────────────
#if DEBUG_LEVEL >= 3
#define LOG_VERBOSE 1
#elif DEBUG_LEVEL >= 1
#define LOG_INFO 1
#else
#define LOG_QUIET 1
#endif

// ── Functions using macro constants ──────────────────────────────────────────

int getBufferSize() {
    return BUFFER_SIZE;
}

int getMaxChannels() {
    return MAX_CHANNELS;
}

int getVersion() {
    return VERSION;
}

int main() {
    printf("SafeC Preprocessor Demo\n");
    printf("Platform    : %s\n", PLATFORM_NAME);
    printf("Version     : %d\n", getVersion());
    printf("Buffer size : %d\n", getBufferSize());
    printf("Max channels: %d\n", getMaxChannels());

#if FEATURE_ENABLED
    printf("Feature     : enabled\n");
#else
    printf("Feature     : disabled\n");
#endif

#if DEBUG_LEVEL == 0
    printf("Log mode    : quiet\n");
#elif DEBUG_LEVEL == 1
    printf("Log mode    : info\n");
#else
    printf("Log mode    : verbose\n");
#endif

    // __FILE__ and __LINE__ are supported (deterministic, reproducible)
    printf("Source file : %s\n", __FILE__);
    printf("This line   : %d\n", __LINE__);

    return 0;
}
