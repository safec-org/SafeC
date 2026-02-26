// SafeC Standard Library — Configurable Logging
// Zero-overhead when LOG_LEVEL = 0.
#pragma once

// Log levels
#define LOG_LEVEL_NONE    0
#define LOG_LEVEL_ERROR   1
#define LOG_LEVEL_WARN    2
#define LOG_LEVEL_INFO    3
#define LOG_LEVEL_DEBUG   4

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

// Backend: fn(level, tag, msg, file, line)
typedef void (*LogBackend)(int level, const char* tag, const char* msg,
                            const char* file, int line);

void log_set_backend(LogBackend backend);
void log_write(int level, const char* tag, const char* msg,
               const char* file, int line);

// Convenience macros (compile out below threshold)
#if LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_E(tag, msg)  log_write(LOG_LEVEL_ERROR, (tag), (msg), __FILE__, __LINE__)
#else
#define LOG_E(tag, msg)  do {} while(0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_WARN
#define LOG_W(tag, msg)  log_write(LOG_LEVEL_WARN, (tag), (msg), __FILE__, __LINE__)
#else
#define LOG_W(tag, msg)  do {} while(0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_I(tag, msg)  log_write(LOG_LEVEL_INFO, (tag), (msg), __FILE__, __LINE__)
#else
#define LOG_I(tag, msg)  do {} while(0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_D(tag, msg)  log_write(LOG_LEVEL_DEBUG, (tag), (msg), __FILE__, __LINE__)
#else
#define LOG_D(tag, msg)  do {} while(0)
#endif
