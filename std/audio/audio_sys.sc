// SafeC Standard Library — audio & MIDI: OS services (see audio_sys.h).
// safeguard: link linux -ldl -lpthread
#pragma once
#include <std/audio/audio_sys.h>

namespace std {

static void audio_sym_missing_() { }

fn void() audio_dlsym_or_(?&AudioLib_ lib, const &char name, &int ok) {
    match (audio_dlsym_(lib, name)) {
        case some(f): { return f; }
        case null: {}
    }
    *ok = 0;
    return audio_sym_missing_;
}

#if defined(__wasm__)
// ════════════════════════════ WebAssembly ════════════════════════════════════
// No dynamic loading, no threads; the clock is the host's performance.now()
// (std/audio/audio_web.js provides 'std_audio_js_now_ms' in the import object).
extern double std_audio_js_now_ms();

?&AudioLib_ audio_dlopen_(const &char name) { return null; }
?&AudioLib_ audio_dlopen2_(const &char name, const ?&char fallback) { return null; }
AudioSym_ audio_dlsym_(?&AudioLib_ lib, const &char name) { return null; }
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
extern AudioLib_* LoadLibraryA(const char* name);       // HMODULE
extern AudioSym_ GetProcAddress(AudioLib_* module, const char* name);   // FARPROC
extern int   QueryPerformanceCounter(long long* out);
extern int   QueryPerformanceFrequency(long long* out);
extern void  Sleep(unsigned int ms);
extern char* getenv(const char* name);
// avrt.dll's AvSetMmThreadCharacteristicsA(LPCSTR task, LPDWORD index) ->
// HANDLE: looked up at run time so std links no avrt.lib.
struct AudioWinHandle_ { char opaque_; };
typedef fn AudioWinHandle_*(const char*, unsigned int*) AvSetMmThreadFn_;

?&AudioLib_ audio_dlopen_(const &char name) {
    unsafe { return (?&AudioLib_)LoadLibraryA(name); }
}
?&AudioLib_ audio_dlopen2_(const &char name, const ?&char fallback) {
    ?&AudioLib_ h = audio_dlopen_(name);
    if (!h.is_null()) return h;
    unsafe {
        if ((const char*)fallback == (const char*)0) return null;
        return (?&AudioLib_)LoadLibraryA((const char*)fallback);
    }
}
AudioSym_ audio_dlsym_(?&AudioLib_ lib, const &char name) {
    match (lib) {
        case some(l): { unsafe { return GetProcAddress((AudioLib_*)l, name); } }
        case null: {}
    }
    return null;
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
        int ok = 1;
        fn void() f = audio_dlsym_or_(audio_dlopen_("avrt.dll"), "AvSetMmThreadCharacteristicsA", &ok);
        if (!ok) return -1;
        AvSetMmThreadFn_ set = (AvSetMmThreadFn_)f;
        unsigned int index = 0U;
        AudioWinHandle_* h = set("Pro Audio", &index);
        return h == (AudioWinHandle_*)0 ? -1 : 0;
    }
}
?&char audio_getenv_(const &char name) {
    unsafe { return (?&char)getenv(name); }
}

#else
// ═════════════════════════════════ POSIX ═════════════════════════════════════
extern AudioLib_* dlopen(const char* file, int mode);
extern AudioSym_ dlsym(AudioLib_* handle, const char* name);
extern char* getenv(const char* name);
extern unsigned long pthread_self();

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
// struct sched_param (+ room).
struct AudioSchedParam_ { int priority; int pad_[3]; };
extern int clock_gettime(int clk, AudioTimespec_* ts);
extern int nanosleep(const AudioTimespec_* req, AudioTimespec_* rem);
extern int pthread_setschedparam(unsigned long thread, int policy, const AudioSchedParam_* param);

?&AudioLib_ audio_dlopen_(const &char name) {
    unsafe { return (?&AudioLib_)dlopen(name, AUDIO_RTLD_NOW); }
}
?&AudioLib_ audio_dlopen2_(const &char name, const ?&char fallback) {
    ?&AudioLib_ h = audio_dlopen_(name);
    if (!h.is_null()) return h;
    unsafe {
        if ((const char*)fallback == (const char*)0) return null;
        return (?&AudioLib_)dlopen((const char*)fallback, AUDIO_RTLD_NOW);
    }
}
AudioSym_ audio_dlsym_(?&AudioLib_ lib, const &char name) {
    match (lib) {
        case some(l): { unsafe { return dlsym((AudioLib_*)l, name); } }
        case null: {}
    }
    return null;
}
long long audio_now_ns_() {
    struct AudioTimespec_ ts;
    ts.tv_sec = (AUDIO_TS_FIELD_)0;
    ts.tv_nsec = (AUDIO_TS_FIELD_)0;
    unsafe { clock_gettime(AUDIO_CLOCK_MONOTONIC_, &ts); }
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}
void audio_sleep_us_(long long us) {
    if (us <= 0LL) return;
    struct AudioTimespec_ ts;
    ts.tv_sec = (AUDIO_TS_FIELD_)(us / 1000000LL);
    ts.tv_nsec = (AUDIO_TS_FIELD_)((us % 1000000LL) * 1000LL);
    unsafe { nanosleep(&ts, (AudioTimespec_*)0); }
}
int audio_thread_make_realtime_(int priority) {
    int p = priority;
    if (p <= 0) p = 70;
    if (p > 99) p = 99;
    struct AudioSchedParam_ param;
    param.priority = p;
    param.pad_[0] = 0; param.pad_[1] = 0; param.pad_[2] = 0;
    int rc = 0;
    unsafe { rc = pthread_setschedparam(pthread_self(), AUDIO_SCHED_FIFO_, &param); }
    return rc == 0 ? 0 : -1;
}
?&char audio_getenv_(const &char name) {
    unsafe { return (?&char)getenv(name); }
}
#endif

} // namespace std
