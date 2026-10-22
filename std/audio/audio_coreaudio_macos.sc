// SafeC Standard Library — audio backend: Core Audio (macOS HAL). See audio.h.
//
// Devices are the HAL's AudioObjects (kAudioHardwarePropertyDevices): name
// ('lnam'), channels per scope (the stream configuration, 'slay'), rates
// (available nominal sample rates, 'nsr#') and the system defaults
// ('dOut' / 'dIn '). A stream installs an AudioDeviceIOProc on the device —
// the HAL calls it on its own real-time thread with the device's buffers
// (one AudioBuffer per device stream, each interleaved float32 — the HAL's
// virtual format) — sets the device's buffer frame size ('fsiz') to the
// requested frames and its nominal rate ('nsrt') to the requested rate,
// and takes hog mode ('oink') with AUDIO_FLAG_EXCLUSIVE. A duplex stream
// must use one device for both directions (an aggregate device, made in
// Audio MIDI Setup, combines two). Processor overloads ('over') are
// reported as AUDIO_STATUS_OUTPUT_UNDERFLOW / _INPUT_OVERFLOW.
//
// Built against the CoreAudio and CoreFoundation frameworks; the C API
// signatures below follow <CoreAudio/AudioHardware.h>
// (AudioObjectID/UInt32 = unsigned int, OSStatus = int, Float64 = double).
// safeguard: link macos -framework CoreAudio -framework CoreFoundation
#pragma once
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

// Four-character codes.
#define CA_SYSTEM_OBJECT        1U
#define CA_PROP_DEVICES         0x64657623U   // 'dev#'
#define CA_PROP_DEFAULT_OUTPUT  0x644F7574U   // 'dOut'
#define CA_PROP_DEFAULT_INPUT   0x64496E20U   // 'dIn '
#define CA_PROP_RUN_LOOP        0x726E6C70U   // 'rnlp'
#define CA_PROP_NAME            0x6C6E616DU   // 'lnam'
#define CA_PROP_STREAM_CONFIG   0x736C6179U   // 'slay'
#define CA_PROP_NOMINAL_RATE    0x6E737274U   // 'nsrt'
#define CA_PROP_AVAILABLE_RATES 0x6E737223U   // 'nsr#'
#define CA_PROP_BUFFER_FRAMES   0x6673697AU   // 'fsiz'
#define CA_PROP_BUFFER_RANGE    0x66737A23U   // 'fsz#'
#define CA_PROP_LATENCY         0x6C746E63U   // 'ltnc'
#define CA_PROP_SAFETY_OFFSET   0x73616674U   // 'saft'
#define CA_PROP_HOG_MODE        0x6F696E6BU   // 'oink'
#define CA_PROP_OVERLOAD        0x6F766572U   // 'over'
#define CA_PROP_IS_ALIVE        0x6C69766EU   // 'livn'
#define CA_SCOPE_GLOBAL         0x676C6F62U   // 'glob'
#define CA_SCOPE_INPUT          0x696E7074U   // 'inpt'
#define CA_SCOPE_OUTPUT         0x6F757470U   // 'outp'
#define CA_ELEMENT_MAIN         0U
#define CA_UTF8                 0x08000100U   // kCFStringEncodingUTF8

namespace std {

struct CAPropertyAddress_ {
    unsigned int selector;
    unsigned int scope;
    unsigned int element;
};

// AudioBuffer / AudioBufferList (64-bit layout: mBuffers at offset 8).
struct CABuffer_ {
    unsigned int channels;
    unsigned int bytes;
    void* data;
};
struct CAValueRange_ {
    double minimum;
    double maximum;
};

static_assert(sizeof(struct CAPropertyAddress_) == 12, "AudioObjectPropertyAddress is 12 bytes");
static_assert(sizeof(struct CABuffer_) == 16, "AudioBuffer is 16 bytes on 64-bit");

extern int  AudioObjectGetPropertyDataSize(unsigned int obj, const CAPropertyAddress_* addr, unsigned int qualSize,
                                           const void* qual, unsigned int* outSize);
extern int  AudioObjectGetPropertyData(unsigned int obj, const CAPropertyAddress_* addr, unsigned int qualSize,
                                       const void* qual, unsigned int* ioSize, void* outData);
extern int  AudioObjectSetPropertyData(unsigned int obj, const CAPropertyAddress_* addr, unsigned int qualSize,
                                       const void* qual, unsigned int size, const void* data);
extern unsigned char AudioObjectHasProperty(unsigned int obj, const CAPropertyAddress_* addr);
extern int  AudioObjectAddPropertyListener(unsigned int obj, const CAPropertyAddress_* addr, void* proc, void* client);
extern int  AudioObjectRemovePropertyListener(unsigned int obj, const CAPropertyAddress_* addr, void* proc, void* client);
extern int  AudioDeviceCreateIOProcID(unsigned int dev, void* proc, void* client, void** outID);
extern int  AudioDeviceDestroyIOProcID(unsigned int dev, void* procID);
extern int  AudioDeviceStart(unsigned int dev, void* procID);
extern int  AudioDeviceStop(unsigned int dev, void* procID);
extern unsigned char CFStringGetCString(const void* str, char* buf, long size, unsigned int encoding);
extern void CFRelease(const void* cf);
extern int  getpid();

static struct CAPropertyAddress_ ca_addr_(unsigned int selector, unsigned int scope) {
    struct CAPropertyAddress_ a;
    a.selector = selector;
    a.scope = scope;
    a.element = CA_ELEMENT_MAIN;
    return a;
}

static int ca_get_u32_(unsigned int obj, unsigned int selector, unsigned int scope, &unsigned int out) {
    struct CAPropertyAddress_ a = ca_addr_(selector, scope);
    unsigned int size = 4U;
    unsafe { return AudioObjectGetPropertyData(obj, &a, 0U, (const void*)0, &size, (void*)out); }
}

static int ca_get_f64_(unsigned int obj, unsigned int selector, unsigned int scope, &double out) {
    struct CAPropertyAddress_ a = ca_addr_(selector, scope);
    unsigned int size = 8U;
    unsafe { return AudioObjectGetPropertyData(obj, &a, 0U, (const void*)0, &size, (void*)out); }
}

// Total channels of a device in one scope (sum over its streams' buffers).
static int ca_channels_(unsigned int dev, unsigned int scope) {
    struct CAPropertyAddress_ a = ca_addr_(CA_PROP_STREAM_CONFIG, scope);
    unsigned int size = 0U;
    int total = 0;
    unsafe {
        if (AudioObjectGetPropertyDataSize(dev, &a, 0U, (const void*)0, &size) != 0 || size < 8U) return 0;
        unsigned char* list = (unsigned char*)alloc_zeroed_raw_((unsigned long)size);
        if (list == (unsigned char*)0) return 0;
        if (AudioObjectGetPropertyData(dev, &a, 0U, (const void*)0, &size, (void*)list) == 0) {
            unsigned int n = *(unsigned int*)list;
            struct CABuffer_* bufs = (struct CABuffer_*)(list + 8);
            for (unsigned int i = 0U; i < n; i = i + 1U) total = total + (int)bufs[i].channels;
        }
        dealloc_raw_((void*)list);
    }
    return total;
}

static int audio_coreaudio_available_() { return 1; }

// Let the HAL run its own notification thread (no run loop of ours).
static void ca_init_run_loop_() {
    struct CAPropertyAddress_ a = ca_addr_(CA_PROP_RUN_LOOP, CA_SCOPE_GLOBAL);
    void* none = (void*)0;
    unsafe { AudioObjectSetPropertyData(CA_SYSTEM_OBJECT, &a, 0U, (const void*)0, (unsigned int)sizeof(void*), (const void*)&none); }
}

static int audio_coreaudio_probe_(&AudioHost_ h) {
    ca_init_run_loop_();
    struct CAPropertyAddress_ a = ca_addr_(CA_PROP_DEVICES, CA_SCOPE_GLOBAL);
    unsigned int size = 0U;
    unsigned int defOut = 0U;
    unsigned int defIn = 0U;
    ca_get_u32_(CA_SYSTEM_OBJECT, CA_PROP_DEFAULT_OUTPUT, CA_SCOPE_GLOBAL, &defOut);
    ca_get_u32_(CA_SYSTEM_OBJECT, CA_PROP_DEFAULT_INPUT, CA_SCOPE_GLOBAL, &defIn);
    unsafe {
        int rc = AudioObjectGetPropertyDataSize(CA_SYSTEM_OBJECT, &a, 0U, (const void*)0, &size);
        if (rc != 0) return audio_fail_code_(&h.error, AUDIO_ERR_SYSTEM, "coreaudio: listing devices failed", (long long)rc);
        if (size == 0U) return AUDIO_OK;
        unsigned int* ids = (unsigned int*)alloc_zeroed_raw_((unsigned long)size);
        if (ids == (unsigned int*)0) return audio_fail_(&h.error, AUDIO_ERR_NO_MEMORY, "coreaudio: out of memory");
        rc = AudioObjectGetPropertyData(CA_SYSTEM_OBJECT, &a, 0U, (const void*)0, &size, (void*)ids);
        unsigned int n = rc == 0 ? size / 4U : 0U;
        for (unsigned int i = 0U; i < n; i = i + 1U) {
            unsigned int dev = ids[i];
            struct AudioDeviceInfo info = audio_device_info_blank_();
            // Name: a CFString.
            struct CAPropertyAddress_ na = ca_addr_(CA_PROP_NAME, CA_SCOPE_GLOBAL);
            void* cfname = (void*)0;
            unsigned int nsize = (unsigned int)sizeof(void*);
            if (AudioObjectGetPropertyData(dev, &na, 0U, (const void*)0, &nsize, (void*)&cfname) == 0 && cfname != (void*)0) {
                CFStringGetCString((const void*)cfname, (char*)&info.name[0], (long)AUDIO_DEVICE_NAME_MAX, CA_UTF8);
                CFRelease((const void*)cfname);
            }
            if (info.name[0] == (char)0) audio_strcpy_(&info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "Core Audio device");
            info.output_channels = ca_channels_(dev, CA_SCOPE_OUTPUT);
            info.input_channels = ca_channels_(dev, CA_SCOPE_INPUT);
            if (info.output_channels == 0 && info.input_channels == 0) continue;
            info.duplex_channels = info.output_channels < info.input_channels ? info.output_channels : info.input_channels;
            info.native_formats = AUDIO_FORMAT_F32;
            // Rates: every standard rate inside an available range.
            struct CAPropertyAddress_ ra = ca_addr_(CA_PROP_AVAILABLE_RATES, CA_SCOPE_GLOBAL);
            unsigned int rsize = 0U;
            if (AudioObjectGetPropertyDataSize(dev, &ra, 0U, (const void*)0, &rsize) == 0 && rsize >= 16U) {
                struct CAValueRange_* ranges = (struct CAValueRange_*)alloc_zeroed_raw_((unsigned long)rsize);
                if (ranges != (struct CAValueRange_*)0) {
                    if (AudioObjectGetPropertyData(dev, &ra, 0U, (const void*)0, &rsize, (void*)ranges) == 0) {
                        unsigned int nr = rsize / 16U;
                        for (unsigned int r = 0U; r < nr; r = r + 1U) {
                            if (ranges[r].minimum == ranges[r].maximum) {
                                audio_device_add_rate_(&info, (unsigned int)ranges[r].minimum);
                                continue;
                            }
                            for (int k = 0; k < audio_standard_rate_count_(); k = k + 1) {
                                double sr = (double)audio_standard_rate_(k);
                                if (sr >= ranges[r].minimum && sr <= ranges[r].maximum)
                                    audio_device_add_rate_(&info, audio_standard_rate_(k));
                            }
                        }
                    }
                    dealloc_raw_((void*)ranges);
                }
            }
            double nominal = 0.0;
            if (ca_get_f64_(dev, CA_PROP_NOMINAL_RATE, CA_SCOPE_GLOBAL, &nominal) == 0 && nominal > 0.0) {
                info.preferred_sample_rate = (unsigned int)nominal;
                audio_device_add_rate_(&info, info.preferred_sample_rate);
            }
            info.is_default_output = dev == defOut && info.output_channels > 0;
            info.is_default_input = dev == defIn && info.input_channels > 0;
            char key[32];
            audio_strcpy_(&key[0], 32UL, "ca:");
            audio_strcat_int_(&key[0], 32UL, (long long)dev);
            match (audio_host_add_(h, &info, &key[0], (long long)dev)) {
                case some(rec): {
                    if (rec.info.is_default_output) h.default_out = rec.info.id;
                    if (rec.info.is_default_input) h.default_in = rec.info.id;
                }
                case null: {}
            }
        }
        dealloc_raw_((void*)ids);
    }
    return AUDIO_OK;
}

static void audio_coreaudio_host_close_(&AudioHost_ h) { }

// ── streams ──────────────────────────────────────────────────────────────

struct CAStream_ {
    unsigned int device;
    void* proc_id;
    int running;            // AudioDeviceStart'ed
    int stopped;            // (rt) the IOProc outputs silence
    int hogged;
    int listening;
    struct AudioStreamState_* st;
};

static ?&CAStream_ ca_of_(&AudioStreamState_ st) {
    unsafe { return (?&CAStream_)st.backend; }
}

// Frames in an AudioBufferList whose first buffer has 'channels'-wide f32 frames.
static unsigned int ca_list_frames_(const unsigned char* list) {
    unsafe {
        if (list == (const unsigned char*)0) return 0U;
        unsigned int n = *(const unsigned int*)list;
        if (n == 0U) return 0U;
        const struct CABuffer_* b = (const struct CABuffer_*)(list + 8);
        if (b[0].channels == 0U) return 0U;
        return b[0].bytes / (b[0].channels * 4U);
    }
}

// The HAL's IOProc (its real-time thread).
static int ca_ioproc_(unsigned int dev, const void* now, const unsigned char* inList, const void* inTime,
                      unsigned char* outList, const void* outTime, void* client) {
    unsafe {
        struct CAStream_* c = (struct CAStream_*)client;
        struct AudioStreamState_* st = c->st;
        int live = atomic_load((_Atomic int*)&st->state) == AUDIO_STREAM_RUNNING &&
                   atomic_load((_Atomic int*)&c->stopped) == 0;
        unsigned int frames = st->mode_out ? ca_list_frames_((const unsigned char*)outList) : ca_list_frames_(inList);
        if (frames > st->max_frames) {
            frames = st->max_frames;
        }
        if (live && st->mode_in && inList != (const unsigned char*)0) {
            // Gather the device's input streams into one interleaved frame.
            float* dst = (float*)st->device_buf[AUDIO_DIR_IN];
            unsigned int total = (unsigned int)st->device_channels[AUDIO_DIR_IN];
            unsigned int nb = *(const unsigned int*)inList;
            const struct CABuffer_* bufs = (const struct CABuffer_*)(inList + 8);
            unsigned int base = 0U;
            for (unsigned int b = 0U; b < nb; b = b + 1U) {
                const float* src = (const float*)bufs[b].data;
                unsigned int ch = bufs[b].channels;
                for (unsigned int f = 0U; f < frames; f = f + 1U)
                    for (unsigned int k = 0U; k < ch && base + k < total; k = k + 1U)
                        dst[f * total + base + k] = src == (const float*)0 ? (float)0.0 : src[f * ch + k];
                base = base + ch;
            }
        }
        int r = AUDIO_CB_CONTINUE;
        if (live) r = audio_stream_process_((&AudioStreamState_)st, frames);
        if (st->mode_out && outList != (unsigned char*)0) {
            unsigned int nb = *(unsigned int*)outList;
            struct CABuffer_* bufs = (struct CABuffer_*)(outList + 8);
            const float* src = (const float*)st->device_buf[AUDIO_DIR_OUT];
            unsigned int total = (unsigned int)st->device_channels[AUDIO_DIR_OUT];
            unsigned int base = 0U;
            for (unsigned int b = 0U; b < nb; b = b + 1U) {
                float* dst = (float*)bufs[b].data;
                unsigned int ch = bufs[b].channels;
                if (dst == (float*)0) { base = base + ch; continue; }
                unsigned int n = bufs[b].bytes / 4U;
                if (!live) {
                    for (unsigned int i = 0U; i < n; i = i + 1U) dst[i] = (float)0.0;
                } else {
                    for (unsigned int f = 0U; f < frames; f = f + 1U)
                        for (unsigned int k = 0U; k < ch; k = k + 1U)
                            dst[f * ch + k] = base + k < total ? src[f * total + base + k] : (float)0.0;
                }
                base = base + ch;
            }
        }
        if (live && r != AUDIO_CB_CONTINUE) {
            // The HAL keeps calling until AudioDeviceStop: go silent and report
            // stopped; stop()/close() stop the device.
            atomic_store((_Atomic int*)&c->stopped, 1);
            audio_stream_mark_stopped_((&AudioStreamState_)st, null);
        }
    }
    return 0;
}

// 'over' (processor overload) and 'livn' (device gone) notifications.
static int ca_listener_(unsigned int obj, unsigned int n, const CAPropertyAddress_* addrs, void* client) {
    unsafe {
        struct CAStream_* c = (struct CAStream_*)client;
        for (unsigned int i = 0U; i < n; i = i + 1U) {
            if (addrs[i].selector == CA_PROP_OVERLOAD)
                audio_stream_flag_status_((&AudioStreamState_)c->st, AUDIO_STATUS_OUTPUT_UNDERFLOW | AUDIO_STATUS_INPUT_OVERFLOW);
            if (addrs[i].selector == CA_PROP_IS_ALIVE) {
                unsigned int alive = 1U;
                ca_get_u32_(obj, CA_PROP_IS_ALIVE, CA_SCOPE_GLOBAL, &alive);
                if (alive == 0U) {
                    atomic_store((_Atomic int*)&c->stopped, 1);
                    audio_stream_set_rt_error_((&AudioStreamState_)c->st, AUDIO_ERR_SYSTEM, "coreaudio: the device disappeared");
                    audio_stream_mark_stopped_((&AudioStreamState_)c->st, null);
                }
            }
        }
    }
    return 0;
}

static void ca_release_(&CAStream_ c) {
    unsafe {
        if (c.running) { AudioDeviceStop(c.device, c.proc_id); c.running = 0; }
        if (c.listening) {
            struct CAPropertyAddress_ o = ca_addr_(CA_PROP_OVERLOAD, CA_SCOPE_GLOBAL);
            struct CAPropertyAddress_ l = ca_addr_(CA_PROP_IS_ALIVE, CA_SCOPE_GLOBAL);
            AudioObjectRemovePropertyListener(c.device, &o, (void*)ca_listener_, (void*)c);
            AudioObjectRemovePropertyListener(c.device, &l, (void*)ca_listener_, (void*)c);
            c.listening = 0;
        }
        if (c.proc_id != (void*)0) { AudioDeviceDestroyIOProcID(c.device, c.proc_id); c.proc_id = (void*)0; }
        if (c.hogged) {
            int none = -1;
            struct CAPropertyAddress_ hog = ca_addr_(CA_PROP_HOG_MODE, CA_SCOPE_GLOBAL);
            AudioObjectSetPropertyData(c.device, &hog, 0U, (const void*)0, 4U, (const void*)&none);
            c.hogged = 0;
        }
    }
}

static int audio_coreaudio_open_(&AudioHost_ h, &AudioStreamState_ st) {
    if (st.mode_out && st.mode_in && st.device_native[AUDIO_DIR_OUT] != st.device_native[AUDIO_DIR_IN])
        return audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED,
                           "coreaudio: duplex needs one device for both directions (use an aggregate device)");
    if (st.device_name[0] != (char)0)
        return audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "coreaudio: device_name is not used; pass a device id");
    unsigned int dev = (unsigned int)(st.mode_out ? st.device_native[AUDIO_DIR_OUT] : st.device_native[AUDIO_DIR_IN]);
    ?&CAStream_ cp = null;
    unsafe { cp = (?&CAStream_)alloc_zeroed_raw_(sizeof(struct CAStream_)); }
    match (cp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "coreaudio: out of memory"); }
        case some(c): {
            c.device = dev;
            unsafe { c.st = (struct AudioStreamState_*)st; }
            int rc = AUDIO_OK;
            unsafe {
                // Exclusive: hog mode.
                if ((st.flags & AUDIO_FLAG_EXCLUSIVE) != 0U) {
                    int pid = getpid();
                    struct CAPropertyAddress_ hog = ca_addr_(CA_PROP_HOG_MODE, CA_SCOPE_GLOBAL);
                    if (AudioObjectSetPropertyData(dev, &hog, 0U, (const void*)0, 4U, (const void*)&pid) == 0) c.hogged = 1;
                    else rc = audio_fail_(&st.error, AUDIO_ERR_DEVICE_BUSY, "coreaudio: could not take hog mode");
                }
                // Sample rate: set it when it differs, then wait for the change.
                double cur = 0.0;
                if (rc == AUDIO_OK && ca_get_f64_(dev, CA_PROP_NOMINAL_RATE, CA_SCOPE_GLOBAL, &cur) == 0 &&
                    (unsigned int)cur != st.sample_rate) {
                    double want = (double)st.sample_rate;
                    struct CAPropertyAddress_ ra = ca_addr_(CA_PROP_NOMINAL_RATE, CA_SCOPE_GLOBAL);
                    int e = AudioObjectSetPropertyData(dev, &ra, 0U, (const void*)0, 8U, (const void*)&want);
                    if (e != 0) rc = audio_fail_code_(&st.error, AUDIO_ERR_INVALID_PARAM, "coreaudio: sample rate not supported", (long long)st.sample_rate);
                    for (int t = 0; rc == AUDIO_OK && t < 100; t = t + 1) {
                        ca_get_f64_(dev, CA_PROP_NOMINAL_RATE, CA_SCOPE_GLOBAL, &cur);
                        if ((unsigned int)cur == st.sample_rate) break;
                        audio_sleep_us_(5000LL);
                    }
                    if (rc == AUDIO_OK && (unsigned int)cur != st.sample_rate)
                        rc = audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "coreaudio: the device didn't switch sample rate");
                }
                // Buffer size: the request, clamped to the device's range.
                if (rc == AUDIO_OK) {
                    unsigned int frames = st.buffer_frames == 0U ? 512U : st.buffer_frames;
                    if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U) frames = 32U;
                    struct CAValueRange_ range;
                    range.minimum = 0.0; range.maximum = 0.0;
                    struct CAPropertyAddress_ fr = ca_addr_(CA_PROP_BUFFER_RANGE, CA_SCOPE_GLOBAL);
                    unsigned int rs = 16U;
                    if (AudioObjectGetPropertyData(dev, &fr, 0U, (const void*)0, &rs, (void*)&range) == 0 && range.maximum > 0.0) {
                        if ((double)frames < range.minimum) frames = (unsigned int)range.minimum;
                        if ((double)frames > range.maximum) frames = (unsigned int)range.maximum;
                    }
                    struct CAPropertyAddress_ fs = ca_addr_(CA_PROP_BUFFER_FRAMES, CA_SCOPE_GLOBAL);
                    AudioObjectSetPropertyData(dev, &fs, 0U, (const void*)0, 4U, (const void*)&frames);
                    unsigned int got = frames;
                    ca_get_u32_(dev, CA_PROP_BUFFER_FRAMES, CA_SCOPE_GLOBAL, &got);
                    st.buffer_frames = got;
                    st.max_frames = got * 2U;     // the HAL may hand a larger cycle after a reconfiguration
                    st.buffer_count = 2U;
                }
                for (int d = 0; d < 2 && rc == AUDIO_OK; d = d + 1) {
                    int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
                    if (!on) continue;
                    unsigned int scope = d == AUDIO_DIR_OUT ? CA_SCOPE_OUTPUT : CA_SCOPE_INPUT;
                    st.device_format[d] = AUDIO_FORMAT_F32;
                    st.device_channels[d] = ca_channels_(dev, scope);
                    st.device_interleaved[d] = 1;
                    unsigned int lat = 0U;
                    unsigned int safety = 0U;
                    ca_get_u32_(dev, CA_PROP_LATENCY, scope, &lat);
                    ca_get_u32_(dev, CA_PROP_SAFETY_OFFSET, scope, &safety);
                    st.latency[d] = (long)lat + (long)safety + (long)st.buffer_frames;
                }
                if (rc == AUDIO_OK) {
                    int e = AudioDeviceCreateIOProcID(dev, (void*)ca_ioproc_, (void*)c, &c.proc_id);
                    if (e != 0) rc = audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coreaudio: AudioDeviceCreateIOProcID", (long long)e);
                }
                if (rc == AUDIO_OK) {
                    struct CAPropertyAddress_ o = ca_addr_(CA_PROP_OVERLOAD, CA_SCOPE_GLOBAL);
                    struct CAPropertyAddress_ l = ca_addr_(CA_PROP_IS_ALIVE, CA_SCOPE_GLOBAL);
                    AudioObjectAddPropertyListener(dev, &o, (void*)ca_listener_, (void*)c);
                    AudioObjectAddPropertyListener(dev, &l, (void*)ca_listener_, (void*)c);
                    c.listening = 1;
                }
            }
            if (rc != AUDIO_OK) {
                ca_release_(c);
                unsafe { dealloc_raw_((void*)c); }
                return rc;
            }
            unsafe { st.backend = (void*)c; }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

static int audio_coreaudio_start_(&AudioStreamState_ st) {
    match (ca_of_(st)) {
        case some(c): {
            unsafe { atomic_store((_Atomic int*)&c.stopped, 0); }
            if (c.running) return AUDIO_OK;
            int e = 0;
            unsafe { e = AudioDeviceStart(c.device, c.proc_id); }
            if (e != 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "coreaudio: AudioDeviceStart", (long long)e);
            c.running = 1;
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "coreaudio: stream not open");
}

static int audio_coreaudio_stop_(&AudioStreamState_ st) {
    match (ca_of_(st)) {
        case some(c): {
            unsafe { atomic_store((_Atomic int*)&c.stopped, 1); }
            if (c.running) {
                // AudioDeviceStop returns once the IOProc is no longer called;
                // the buffer it last filled has been handed to the device.
                unsafe { AudioDeviceStop(c.device, c.proc_id); }
                c.running = 0;
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_OK;
}

static void audio_coreaudio_close_(&AudioStreamState_ st) {
    match (ca_of_(st)) {
        case some(c): {
            ca_release_(c);
            unsafe { dealloc_raw_((void*)c); }
        }
        case null: {}
    }
    unsafe { st.backend = (void*)0; }
}

static struct AudioBackendVt_ g_audio_coreaudio_vt_;
static int g_audio_coreaudio_vt_ready_ = 0;

&static AudioBackendVt_ audio_coreaudio_backend_() {
    if (!g_audio_coreaudio_vt_ready_) {
        g_audio_coreaudio_vt_.api = AUDIO_API_COREAUDIO;
        g_audio_coreaudio_vt_.available = audio_coreaudio_available_;
        g_audio_coreaudio_vt_.probe = audio_coreaudio_probe_;
        g_audio_coreaudio_vt_.host_close = audio_coreaudio_host_close_;
        g_audio_coreaudio_vt_.open = audio_coreaudio_open_;
        g_audio_coreaudio_vt_.start = audio_coreaudio_start_;
        g_audio_coreaudio_vt_.stop = audio_coreaudio_stop_;
        g_audio_coreaudio_vt_.abort = audio_coreaudio_stop_;
        g_audio_coreaudio_vt_.close = audio_coreaudio_close_;
        g_audio_coreaudio_vt_ready_ = 1;
    }
    return &g_audio_coreaudio_vt_;
}

} // namespace std
