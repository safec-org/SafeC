// SafeC Standard Library — audio & MIDI: AudioError and the small string
// helpers std/audio uses (see audio_error.h).
#pragma once
#include <std/audio/audio_error.h>

namespace std {

unsigned long audio_strlen_(const ?&char s) {
    unsigned long n = 0UL;
    unsafe {
        const char* p = (const char*)s;
        if (p == (const char*)0) return 0UL;
        // Bounded, so LLVM doesn't turn it into a strlen() call: wasm32
        // modules have no libc to import one from.
        while (n < 0x7FFFFFFFUL && p[n] != (char)0) n = n + 1UL;
    }
    return n;
}

void audio_strcpy_(&char dst, unsigned long cap, const ?&char src) {
    if (cap == 0UL) return;
    unsafe {
        char* d = (char*)dst;
        const char* s = (const char*)src;
        unsigned long i = 0UL;
        if (s != (const char*)0) {
            while (i + 1UL < cap && s[i] != (char)0) { d[i] = s[i]; i = i + 1UL; }
        }
        d[i] = (char)0;
    }
}

void audio_strcat_(&char dst, unsigned long cap, const ?&char src) {
    unsigned long n = audio_strlen_(dst);
    if (n + 1UL >= cap) return;
    unsafe {
        char* d = (char*)dst;
        audio_strcpy_((&char)(d + n), cap - n, src);
    }
}

int audio_streq_(const ?&char a, const ?&char b) {
    unsafe {
        const char* x = (const char*)a;
        const char* y = (const char*)b;
        if (x == (const char*)0 || y == (const char*)0) return x == y;
        unsigned long i = 0UL;
        while (x[i] != (char)0 && x[i] == y[i]) i = i + 1UL;
        return x[i] == y[i];
    }
}

int audio_strprefix_(const ?&char s, const ?&char prefix) {
    unsafe {
        const char* x = (const char*)s;
        const char* p = (const char*)prefix;
        if (x == (const char*)0 || p == (const char*)0) return 0;
        unsigned long i = 0UL;
        while (p[i] != (char)0) {
            if (x[i] != p[i]) return 0;
            i = i + 1UL;
        }
        return 1;
    }
}

void audio_strcat_int_(&char dst, unsigned long cap, long long v) {
    char tmp[24];
    int n = 0;
    int neg = v < 0LL;
    unsigned long long u = neg ? (unsigned long long)(0LL - v) : (unsigned long long)v;
    if (u == 0ULL) { tmp[0] = '0'; n = 1; }
    while (u > 0ULL && n < 22) { tmp[n] = (char)(48 + (int)(u % 10ULL)); u = u / 10ULL; n = n + 1; }
    char out[24];
    int k = 0;
    if (neg) { out[0] = '-'; k = 1; }
    while (n > 0) { n = n - 1; out[k] = tmp[n]; k = k + 1; }
    out[k] = (char)0;
    audio_strcat_(dst, cap, &out[0]);
}

struct AudioError AudioError::none() {
    struct AudioError e;
    e.code = AUDIO_OK;
    e.message[0] = (char)0;
    return e;
}

struct AudioError AudioError::make(int code, const &char message) {
    struct AudioError e;
    e.code = code;
    audio_strcpy_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, message);
    return e;
}

struct AudioError AudioError::make2(int code, const &char prefix, const &char detail) {
    struct AudioError e;
    e.code = code;
    audio_strcpy_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, prefix);
    audio_strcat_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, ": ");
    audio_strcat_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, detail);
    return e;
}

struct AudioError AudioError::make_code(int code, const &char prefix, long long value) {
    struct AudioError e;
    e.code = code;
    audio_strcpy_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, prefix);
    audio_strcat_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, " (");
    audio_strcat_int_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, value);
    audio_strcat_(&e.message[0], (unsigned long)AUDIO_ERROR_MESSAGE_MAX, ")");
    return e;
}

int AudioError::ok() const { return self.code == AUDIO_OK || self.code == AUDIO_ERR_WARNING; }
int AudioError::failed() const { return !(self.code == AUDIO_OK || self.code == AUDIO_ERR_WARNING); }

int audio_fail_(&AudioError e, int code, const &char message) {
    *e = AudioError::make(code, message);
    return code;
}
int audio_fail2_(&AudioError e, int code, const &char prefix, const &char detail) {
    *e = AudioError::make2(code, prefix, detail);
    return code;
}
int audio_fail_code_(&AudioError e, int code, const &char prefix, long long value) {
    *e = AudioError::make_code(code, prefix, value);
    return code;
}

const &char audio_error_name(int code) {
    if (code == AUDIO_OK) return "ok";
    if (code == AUDIO_ERR_WARNING) return "warning";
    if (code == AUDIO_ERR_UNSUPPORTED) return "unsupported";
    if (code == AUDIO_ERR_NO_DEVICES) return "no devices";
    if (code == AUDIO_ERR_INVALID_DEVICE) return "invalid device";
    if (code == AUDIO_ERR_INVALID_PARAM) return "invalid parameter";
    if (code == AUDIO_ERR_INVALID_USE) return "invalid use";
    if (code == AUDIO_ERR_DEVICE_BUSY) return "device busy";
    if (code == AUDIO_ERR_SYSTEM) return "system error";
    if (code == AUDIO_ERR_THREAD) return "thread error";
    if (code == AUDIO_ERR_NO_MEMORY) return "out of memory";
    if (code == AUDIO_ERR_UNAVAILABLE) return "unavailable";
    return "unknown error";
}

} // namespace std
