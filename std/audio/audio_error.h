// SafeC Standard Library — audio & MIDI: the error value both APIs return.
//
// std/audio reports failures the way the rest of std does (no exceptions):
// an operation returns an AudioError — 'code' is AUDIO_OK (0) on success or
// one of the AUDIO_ERR_* codes below, 'message' a human-readable line for
// the log (the backend's own error text where it has one: ALSA's
// snd_strerror(), an HRESULT, a Core Audio OSStatus) — or, when it also
// produces a value, a std::Result<T, AudioError> (see std/result.h):
//
//   struct Result<Audio, AudioError> r = std::Audio::open(AUDIO_API_UNSPECIFIED);
//   match (r.get_err()) { case some(e): printf("%s\n", e.message); case null: {} }
//
// Shared by audio.h and midi.h.
#pragma once

#define AUDIO_OK                  0
#define AUDIO_ERR_WARNING         1   // non-fatal (e.g. realtime priority refused)
#define AUDIO_ERR_UNSUPPORTED     2   // the API isn't compiled in / can't do this
#define AUDIO_ERR_NO_DEVICES      3   // no (default) device to use
#define AUDIO_ERR_INVALID_DEVICE  4   // no device/port with that id/index
#define AUDIO_ERR_INVALID_PARAM   5   // a parameter the API can't accept
#define AUDIO_ERR_INVALID_USE     6   // wrong state (start an unopened stream, ...)
#define AUDIO_ERR_DEVICE_BUSY     7   // the device is in use (exclusive mode)
#define AUDIO_ERR_SYSTEM          8   // the OS / library call failed
#define AUDIO_ERR_THREAD          9   // couldn't start the stream thread
#define AUDIO_ERR_NO_MEMORY      10   // allocation failed (open time only)
#define AUDIO_ERR_UNAVAILABLE    11   // the backend's library/server isn't there

#define AUDIO_ERROR_MESSAGE_MAX 192

namespace std {

struct AudioError {
    int  code;                              // AUDIO_OK or AUDIO_ERR_*
    char message[AUDIO_ERROR_MESSAGE_MAX];  // NUL-terminated, "" when ok

    // AUDIO_OK, empty message.
    static struct AudioError none();
    // 'code' with 'message' (truncated to fit).
    static struct AudioError make(int code, const &char message);
    // 'code' with "<prefix>: <detail>" — the usual "what failed: why".
    static struct AudioError make2(int code, const &char prefix, const &char detail);
    // 'code' with "<prefix> (<value>)", e.g. an HRESULT or OSStatus.
    static struct AudioError make_code(int code, const &char prefix, long long value);

    int ok() const;        // 1 when code is AUDIO_OK or AUDIO_ERR_WARNING
    int failed() const;    // !ok()
};

// The fixed English name of an AUDIO_ERR_* code ("unsupported", ...).
const &char audio_error_name(int code);

// Bounded string helpers std/audio uses for device and port names (no libc
// on wasm32). Copy 'src' into 'dst' (capacity 'cap', always NUL-terminated);
// append; length; equality.
void          audio_strcpy_(&char dst, unsigned long cap, const ?&char src);
void          audio_strcat_(&char dst, unsigned long cap, const ?&char src);
unsigned long audio_strlen_(const ?&char s);
int           audio_streq_(const ?&char a, const ?&char b);
// Appends the decimal 'v'.
void          audio_strcat_int_(&char dst, unsigned long cap, long long v);

} // namespace std
