// SafeC Standard Library — audio & MIDI: OS services (see audio_sys.h).
// safeguard: link linux -ldl -lpthread
#pragma once
#include <std/audio/audio_sys.h>

namespace std {

#if defined(__wasm__)
// ════════════════════════════ WebAssembly ════════════════════════════════════
// No dynamic loading, no threads; the clock is the host's performance.now()
// (std/audio/audio_web.js provides 'std_audio_js_now_ms' in the import object).
extern double std_audio_js_now_ms();

?&void audio_dlopen_(const &char name) { return null; }
?&void audio_dlopen2_(const &char name, const ?&char fallback) { return null; }
?&void audio_dlsym_(?&void lib, const &char name) { return null; }
long long audio_now_ns_() {
    double ms = 0.0;
    unsafe { ms = std_audio_js_now_ms(); }
    return (long long)(ms * 1000000.0);
}
void audio_sleep_us_(long long us) { }
int audio_thread_make_realtime_(int priority) { return -1; }
?&char audio_getenv_(const &char name) { return null; }

#elif defined(_WIN32)
// ════════════════════════════════ Windows ════════════════════════════════════
extern void* LoadLibraryA(const char* name);
extern void* GetProcAddress(void* module, const char* name);
extern int   QueryPerformanceCounter(long long* out);
extern int   QueryPerformanceFrequency(long long* out);
extern void  Sleep(unsigned int ms);
extern char* getenv(const char* name);
// avrt.dll's AvSetMmThreadCharacteristicsA(LPCSTR task, LPDWORD index) ->
// HANDLE: looked up at run time so std links no avrt.lib.
typedef fn void*(const char*, unsigned int*) AvSetMmThreadFn_;

?&void audio_dlopen_(const &char name) {
    unsafe { return (?&void)LoadLibraryA(name); }
}
?&void audio_dlopen2_(const &char name, const ?&char fallback) {
    ?&void h = audio_dlopen_(name);
    if (!h.is_null()) return h;
    unsafe {
        if ((const char*)fallback == (const char*)0) return null;
        return (?&void)LoadLibraryA((const char*)fallback);
    }
}
?&void audio_dlsym_(?&void lib, const &char name) {
    unsafe {
        if ((void*)lib == (void*)0) return null;
        return (?&void)GetProcAddress((void*)lib, name);
    }
}
long long audio_now_ns_() {
    long long freq = 0LL;
    long long counter = 0LL;
    unsafe {
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);
    }
    if (freq <= 0LL) return 0LL;
    long long whole = counter / freq;
    long long rem = counter % freq;
    return whole * 1000000000LL + (rem * 1000000000LL) / freq;
}
void audio_sleep_us_(long long us) {
    long long ms = (us + 999LL) / 1000LL;
    if (ms < 0LL) ms = 0LL;
    unsafe { Sleep((unsigned int)ms); }
}
int audio_thread_make_realtime_(int priority) {
    unsafe {
        void* avrt = LoadLibraryA("avrt.dll");
        if (avrt == (void*)0) return -1;
        void* f = GetProcAddress(avrt, "AvSetMmThreadCharacteristicsA");
        if (f == (void*)0) return -1;
        AvSetMmThreadFn_ set = (AvSetMmThreadFn_)f;
        unsigned int index = 0U;
        void* h = set("Pro Audio", &index);
        return h == (void*)0 ? -1 : 0;
    }
}
?&char audio_getenv_(const &char name) {
    unsafe { return (?&char)getenv(name); }
}

#else
// ═════════════════════════════════ POSIX ═════════════════════════════════════
extern void* dlopen(const char* file, int mode);
extern void* dlsym(void* handle, const char* name);
extern char* getenv(const char* name);
extern int   clock_gettime(int clk, void* ts);
extern int   nanosleep(const void* req, void* rem);
extern unsigned long pthread_self();
extern int   pthread_setschedparam(unsigned long thread, int policy, const void* param);

#define AUDIO_RTLD_NOW 2
#if defined(__APPLE__)
#define AUDIO_CLOCK_MONOTONIC_ 6
#define AUDIO_SCHED_FIFO_      4
#else
#define AUDIO_CLOCK_MONOTONIC_ 1
#define AUDIO_SCHED_FIFO_      1
#endif
#if __SIZEOF_POINTER__ == 4
#define AUDIO_TS_FIELD_ int
#else
#define AUDIO_TS_FIELD_ long long
#endif
struct AudioTimespec_ {
    AUDIO_TS_FIELD_ tv_sec;
    AUDIO_TS_FIELD_ tv_nsec;
};

?&void audio_dlopen_(const &char name) {
    unsafe { return (?&void)dlopen(name, AUDIO_RTLD_NOW); }
}
?&void audio_dlopen2_(const &char name, const ?&char fallback) {
    ?&void h = audio_dlopen_(name);
    if (!h.is_null()) return h;
    unsafe {
        if ((const char*)fallback == (const char*)0) return null;
        return (?&void)dlopen((const char*)fallback, AUDIO_RTLD_NOW);
    }
}
?&void audio_dlsym_(?&void lib, const &char name) {
    unsafe {
        if ((void*)lib == (void*)0) return null;
        return (?&void)dlsym((void*)lib, name);
    }
}
long long audio_now_ns_() {
    struct AudioTimespec_ ts;
    ts.tv_sec = (AUDIO_TS_FIELD_)0;
    ts.tv_nsec = (AUDIO_TS_FIELD_)0;
    unsafe { clock_gettime(AUDIO_CLOCK_MONOTONIC_, (void*)&ts); }
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}
void audio_sleep_us_(long long us) {
    if (us <= 0LL) return;
    struct AudioTimespec_ ts;
    ts.tv_sec = (AUDIO_TS_FIELD_)(us / 1000000LL);
    ts.tv_nsec = (AUDIO_TS_FIELD_)((us % 1000000LL) * 1000LL);
    unsafe { nanosleep((const void*)&ts, (void*)0); }
}
int audio_thread_make_realtime_(int priority) {
    int p = priority;
    if (p <= 0) p = 70;
    if (p > 99) p = 99;
    int param[4];
    param[0] = p; param[1] = 0; param[2] = 0; param[3] = 0;   // struct sched_param (+ room)
    int rc = 0;
    unsafe { rc = pthread_setschedparam(pthread_self(), AUDIO_SCHED_FIFO_, (const void*)&param[0]); }
    return rc == 0 ? 0 : -1;
}
?&char audio_getenv_(const &char name) {
    unsafe { return (?&char)getenv(name); }
}
#endif

} // namespace std
