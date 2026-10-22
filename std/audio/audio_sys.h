// SafeC Standard Library — audio & MIDI: the OS services the backends share.
//
// std-private (the '_' names): dynamic loading of the backends' system
// libraries (ALSA's libasound and JACK's libjack are dlopen()ed so a build
// runs on machines without them), a monotonic clock, sleeping, real-time
// thread priority and environment lookups — one implementation per OS in
// audio_sys.sc, with the WebAssembly host's clock imported from the JS glue
// (std/audio/audio_web.js).
#pragma once

namespace std {

// dlopen()/LoadLibraryA: a library handle, or null if it isn't installed.
// The names are tried in order.
?&void audio_dlopen_(const &char name);
?&void audio_dlopen2_(const &char name, const ?&char fallback);
// dlsym()/GetProcAddress: null if the library lacks it.
?&void audio_dlsym_(?&void lib, const &char name);

// Monotonic nanoseconds (arbitrary origin).
long long audio_now_ns_();
// Sleeps the calling thread (no-op on wasm32).
void audio_sleep_us_(long long us);

// Raises the calling thread to real-time scheduling: SCHED_FIFO with
// 'priority' (1..99, 0 = a sensible default) on POSIX, the MMCSS "Pro
// Audio" task on Windows. 0 on success; failure (no permission) is not an
// error for the stream, just for the flag.
int audio_thread_make_realtime_(int priority);

// getenv(); null when unset or on wasm32.
?&char audio_getenv_(const &char name);

} // namespace std
