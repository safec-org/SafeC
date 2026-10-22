// SafeC Standard Library — audio backend: WASAPI (Windows). See audio.h.
//
// Devices are the active render and capture endpoints (IMMDeviceEnumerator::
// EnumAudioEndpoints): one device per endpoint, output-only or input-only,
// named by PKEY_Device_FriendlyName; channels and the preferred rate come
// from the shared-mode mix format, the rate list from what exclusive mode
// accepts. A duplex stream pairs a render endpoint with a capture one.
//
// Shared mode (the default): the engine's mix format (normally float32 at
// the mix rate, all of the endpoint's channels); another rate is resampled
// by the engine (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM). AUDIO_FLAG_EXCLUSIVE:
// exclusive mode in the first of the stream's format, f32, i32, i24, i16
// the device takes, the buffer re-aligned when the driver asks
// (AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED). Both are event-driven: one stream
// thread std starts (registered with MMCSS as "Pro Audio") waits on the
// render endpoint's event — the capture one's for input-only streams —
// runs the callback in fixed periods of the stream's buffer size, and
// queues capture packets in a FIFO in between so duplex periods line up.
// Data discontinuities and an empty render buffer are reported as
// AUDIO_STATUS_INPUT_OVERFLOW / _OUTPUT_UNDERFLOW.
//
// COM is used through explicit vtables (index = method order in the
// interface, after IUnknown's QueryInterface/AddRef/Release); the
// interfaces, GUIDs and structures follow <mmdeviceapi.h>, <audioclient.h>,
// <mmreg.h> and <propidl.h> for x64/arm64 (LLP64: DWORD = unsigned int,
// REFERENCE_TIME = long long, HRESULT = int).
// safeguard: link windows -lole32
#pragma once
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define W_COINIT_MULTITHREADED 0U
#define W_CLSCTX_ALL           0x17U
#define W_E_RENDER             0
#define W_E_CAPTURE            1
#define W_E_CONSOLE            0
#define W_DEVICE_STATE_ACTIVE  1U
#define W_STGM_READ            0U
#define W_VT_LPWSTR            31
#define W_SHAREMODE_SHARED     0
#define W_SHAREMODE_EXCLUSIVE  1
#define W_FLAG_EVENTCALLBACK   0x00040000U
#define W_FLAG_AUTOCONVERTPCM  0x80000000U
#define W_FLAG_SRC_DEFAULT_QUALITY 0x08000000U
#define W_BUFFERFLAGS_DISCONTINUITY 1U
#define W_BUFFERFLAGS_SILENT   2U
#define W_FORMAT_PCM           1
#define W_FORMAT_IEEE_FLOAT    3
#define W_FORMAT_EXTENSIBLE    0xFFFE
#define W_CP_UTF8              65001U
#define W_WAIT_TIMEOUT         0x102U
#define W_E_NOT_ALIGNED        (0x88890019U)
#define W_E_DEVICE_IN_USE      (0x8889000AU)

namespace std {

struct WGuid_ {
    unsigned int   d1;
    unsigned short d2;
    unsigned short d3;
    unsigned char  d4[8];
};

// WAVEFORMATEXTENSIBLE (40 bytes; its WAVEFORMATEX head is the first 18).
struct WFormat_ {
    unsigned short tag;
    unsigned short channels;
    unsigned int   rate;
    unsigned int   avg_bytes;
    unsigned short block_align;
    unsigned short bits;
    unsigned short cb_size;
    unsigned short valid_bits;
    unsigned int   channel_mask;
    struct WGuid_  sub;
};

struct WPropKey_ {
    struct WGuid_ fmtid;
    unsigned int  pid;
};

// PROPVARIANT (24 bytes on 64-bit Windows).
struct WPropVariant_ {
    unsigned short vt;
    unsigned short r1;
    unsigned short r2;
    unsigned short r3;
    void*          p;
    unsigned long long pad;
};

static_assert(sizeof(struct WGuid_) == 16, "GUID is 16 bytes");
static_assert(sizeof(struct WFormat_) == 40, "WAVEFORMATEXTENSIBLE is 40 bytes");
static_assert(sizeof(struct WPropVariant_) == 24, "PROPVARIANT is 24 bytes on 64-bit Windows");

extern int   CoInitializeEx(void* reserved, unsigned int coinit);
extern int   CoCreateInstance(const WGuid_* clsid, void* outer, unsigned int ctx, const WGuid_* iid, void** out);
extern void  CoTaskMemFree(void* p);
extern int   PropVariantClear(void* pv);
extern void* CreateEventA(void* attrs, int manualReset, int initialState, const char* name);
extern unsigned int WaitForSingleObject(void* h, unsigned int ms);
extern int   CloseHandle(void* h);
extern int   WideCharToMultiByte(unsigned int cp, unsigned int flags, const unsigned short* w, int wlen,
                                 char* out, int outlen, const char* defChar, int* usedDef);
extern int   MultiByteToWideChar(unsigned int cp, unsigned int flags, const char* s, int slen,
                                 unsigned short* out, int outlen);

// ── COM plumbing ─────────────────────────────────────────────────────────

static struct WGuid_ w_guid_(unsigned int d1, unsigned short d2, unsigned short d3, unsigned char a,
                             unsigned char b, unsigned char c, unsigned char d, unsigned char e,
                             unsigned char f, unsigned char g, unsigned char h) {
    struct WGuid_ x;
    x.d1 = d1; x.d2 = d2; x.d3 = d3;
    x.d4[0] = a; x.d4[1] = b; x.d4[2] = c; x.d4[3] = d;
    x.d4[4] = e; x.d4[5] = f; x.d4[6] = g; x.d4[7] = h;
    return x;
}

// {BCDE0395-E52F-467C-8E3D-C4579291692E}
static struct WGuid_ w_clsid_enumerator_() {
    return w_guid_(0xBCDE0395U, (unsigned short)0xE52F, (unsigned short)0x467C, (unsigned char)0x8E, (unsigned char)0x3D,
                   (unsigned char)0xC4, (unsigned char)0x57, (unsigned char)0x92, (unsigned char)0x91, (unsigned char)0x69, (unsigned char)0x2E);
}
// {A95664D2-9614-4F35-A746-DE8DB63617E6}
static struct WGuid_ w_iid_enumerator_() {
    return w_guid_(0xA95664D2U, (unsigned short)0x9614, (unsigned short)0x4F35, (unsigned char)0xA7, (unsigned char)0x46,
                   (unsigned char)0xDE, (unsigned char)0x8D, (unsigned char)0xB6, (unsigned char)0x36, (unsigned char)0x17, (unsigned char)0xE6);
}
// {1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}
static struct WGuid_ w_iid_audio_client_() {
    return w_guid_(0x1CB9AD4CU, (unsigned short)0xDBFA, (unsigned short)0x4C32, (unsigned char)0xB1, (unsigned char)0x78,
                   (unsigned char)0xC2, (unsigned char)0xF5, (unsigned char)0x68, (unsigned char)0xA7, (unsigned char)0x03, (unsigned char)0xB2);
}
// {F294ACFC-3146-4483-A7BF-ADDCA7C260E2}
static struct WGuid_ w_iid_render_client_() {
    return w_guid_(0xF294ACFCU, (unsigned short)0x3146, (unsigned short)0x4483, (unsigned char)0xA7, (unsigned char)0xBF,
                   (unsigned char)0xAD, (unsigned char)0xDC, (unsigned char)0xA7, (unsigned char)0xC2, (unsigned char)0x60, (unsigned char)0xE2);
}
// {C8ADBD64-E71E-48A0-A4DE-185C395CD317}
static struct WGuid_ w_iid_capture_client_() {
    return w_guid_(0xC8ADBD64U, (unsigned short)0xE71E, (unsigned short)0x48A0, (unsigned char)0xA4, (unsigned char)0xDE,
                   (unsigned char)0x18, (unsigned char)0x5C, (unsigned char)0x39, (unsigned char)0x5C, (unsigned char)0xD3, (unsigned char)0x17);
}
// KSDATAFORMAT_SUBTYPE_PCM {00000001-0000-0010-8000-00AA00389B71}; IEEE_FLOAT is ...0003-...
static struct WGuid_ w_subtype_(unsigned int d1) {
    return w_guid_(d1, (unsigned short)0x0000, (unsigned short)0x0010, (unsigned char)0x80, (unsigned char)0x00,
                   (unsigned char)0x00, (unsigned char)0xAA, (unsigned char)0x00, (unsigned char)0x38, (unsigned char)0x9B, (unsigned char)0x71);
}
// PKEY_Device_FriendlyName {A45C254E-DF1C-4EFD-8020-67D146A850E0}, 14
static struct WPropKey_ w_pkey_friendly_name_() {
    struct WPropKey_ k;
    k.fmtid = w_guid_(0xA45C254EU, (unsigned short)0xDF1C, (unsigned short)0x4EFD, (unsigned char)0x80, (unsigned char)0x20,
                      (unsigned char)0x67, (unsigned char)0xD1, (unsigned char)0x46, (unsigned char)0xA8, (unsigned char)0x50, (unsigned char)0xE0);
    k.pid = 14U;
    return k;
}

static void* w_method_(void* obj, int index) {
    unsafe {
        void** vtbl = *(void***)obj;
        return vtbl[index];
    }
}

typedef fn unsigned int(void*) WRelease_;                                   // IUnknown::Release (2)
// IMMDeviceEnumerator
typedef fn int(void*, int, unsigned int, void**) WEnumEndpoints_;           // 3 EnumAudioEndpoints
typedef fn int(void*, int, int, void**) WGetDefault_;                       // 4 GetDefaultAudioEndpoint
typedef fn int(void*, const unsigned short*, void**) WGetDevice_;           // 5 GetDevice
// IMMDeviceCollection
typedef fn int(void*, unsigned int*) WCollCount_;                           // 3 GetCount
typedef fn int(void*, unsigned int, void**) WCollItem_;                     // 4 Item
// IMMDevice
typedef fn int(void*, const WGuid_*, unsigned int, void*, void**) WActivate_;   // 3 Activate
typedef fn int(void*, unsigned int, void**) WOpenProps_;                    // 4 OpenPropertyStore
typedef fn int(void*, unsigned short**) WGetId_;                            // 5 GetId
// IPropertyStore
typedef fn int(void*, const WPropKey_*, WPropVariant_*) WGetValue_;         // 5 GetValue
// IAudioClient
typedef fn int(void*, int, unsigned int, long long, long long, const WFormat_*, const WGuid_*) WInitialize_;  // 3
typedef fn int(void*, unsigned int*) WGetBufferSize_;                      // 4 GetBufferSize, 6 GetCurrentPadding
typedef fn int(void*, long long*) WGetLatency_;                             // 5 GetStreamLatency
typedef fn int(void*, int, const WFormat_*, void**) WIsSupported_;          // 7 IsFormatSupported
typedef fn int(void*, void**) WGetMixFormat_;                               // 8 GetMixFormat
typedef fn int(void*, long long*, long long*) WGetPeriod_;                  // 9 GetDevicePeriod
typedef fn int(void*) WClientNoArg_;                                        // 10 Start, 11 Stop, 12 Reset
typedef fn int(void*, void*) WSetEvent_;                                    // 13 SetEventHandle
typedef fn int(void*, const WGuid_*, void**) WGetService_;                  // 14 GetService
// IAudioRenderClient
typedef fn int(void*, unsigned int, unsigned char**) WRenderGet_;          // 3 GetBuffer
typedef fn int(void*, unsigned int, unsigned int) WRenderRelease_;          // 4 ReleaseBuffer
// IAudioCaptureClient
typedef fn int(void*, unsigned char**, unsigned int*, unsigned int*, unsigned long long*, unsigned long long*) WCaptureGet_; // 3
typedef fn int(void*, unsigned int) WCaptureRelease_;                       // 4 ReleaseBuffer
typedef fn int(void*, unsigned int*) WCaptureNext_;                         // 5 GetNextPacketSize

static void w_release_(void* obj) {
    unsafe {
        if (obj == (void*)0) return;
        WRelease_ f = (WRelease_)w_method_(obj, 2);
        f(obj);
    }
}

static int w_coinit_() {
    int hr = 0;
    unsafe { hr = CoInitializeEx((void*)0, W_COINIT_MULTITHREADED); }
    // S_OK, S_FALSE (already), RPC_E_CHANGED_MODE (an STA thread: still usable).
    return hr >= 0 || (unsigned int)hr == 0x80010106U;
}

static void* w_enumerator_(&int hrOut) {
    void* en = (void*)0;
    struct WGuid_ clsid = w_clsid_enumerator_();
    struct WGuid_ iid = w_iid_enumerator_();
    unsafe { *hrOut = CoCreateInstance(&clsid, (void*)0, W_CLSCTX_ALL, &iid, &en); }
    return *hrOut < 0 ? (void*)0 : en;
}

// A wide string (UTF-16) to UTF-8 'out'.
static void w_to_utf8_(const unsigned short* w, &char out, unsigned long cap) {
    unsafe {
        out[0] = (char)0;
        if (w == (const unsigned short*)0) return;
        int n = WideCharToMultiByte(W_CP_UTF8, 0U, w, -1, (char*)out, (int)cap, (const char*)0, (int*)0);
        if (n <= 0) out[0] = (char)0;
    }
}

static int w_audio_client_(void* device, &void* out) {
    struct WGuid_ iid = w_iid_audio_client_();
    unsafe {
        WActivate_ act = (WActivate_)w_method_(device, 3);
        return act(device, &iid, W_CLSCTX_ALL, (void*)0, (void**)out);
    }
}

// The AUDIO_FORMAT_* of a WAVEFORMAT(EXTENSIBLE), 0 if none.
static unsigned int w_format_of_(const WFormat_* f) {
    unsafe {
        int isFloat = (int)f->tag == W_FORMAT_IEEE_FLOAT;
        int isPcm = (int)f->tag == W_FORMAT_PCM;
        if ((int)f->tag == W_FORMAT_EXTENSIBLE && f->cb_size >= (unsigned short)22) {
            isFloat = f->sub.d1 == 3U;
            isPcm = f->sub.d1 == 1U;
        }
        if (isFloat && f->bits == (unsigned short)32) return AUDIO_FORMAT_F32;
        if (isFloat && f->bits == (unsigned short)64) return AUDIO_FORMAT_F64;
        if (isPcm && f->bits == (unsigned short)16) return AUDIO_FORMAT_I16;
        if (isPcm && f->bits == (unsigned short)24) return AUDIO_FORMAT_I24;
        if (isPcm && f->bits == (unsigned short)32) return AUDIO_FORMAT_I32;
        if (isPcm && f->bits == (unsigned short)8) return AUDIO_FORMAT_I8;
    }
    return 0U;
}

static struct WFormat_ w_make_format_(unsigned int format, int channels, unsigned int rate) {
    struct WFormat_ f;
    unsafe { safe_memset_raw_((void*)&f, 0, sizeof(struct WFormat_)); }
    int bytes = audio_format_bytes(format);
    f.tag = (unsigned short)W_FORMAT_EXTENSIBLE;
    f.channels = (unsigned short)channels;
    f.rate = rate;
    f.bits = (unsigned short)(bytes * 8);
    f.block_align = (unsigned short)(bytes * channels);
    f.avg_bytes = rate * (unsigned int)(bytes * channels);
    f.cb_size = (unsigned short)22;
    f.valid_bits = f.bits;
    f.channel_mask = channels == 1 ? 0x4U : (channels == 2 ? 0x3U : 0U);
    f.sub = w_subtype_(format == AUDIO_FORMAT_F32 || format == AUDIO_FORMAT_F64 ? 3U : 1U);
    return f;
}

static int audio_wasapi_available_() { return 1; }

static void w_probe_flow_(&AudioHost_ h, void* en, int flow, &char defId) {
    unsafe {
        void* coll = (void*)0;
        WEnumEndpoints_ enumEp = (WEnumEndpoints_)w_method_(en, 3);
        if (enumEp(en, flow, W_DEVICE_STATE_ACTIVE, &coll) < 0 || coll == (void*)0) return;
        unsigned int count = 0U;
        WCollCount_ getCount = (WCollCount_)w_method_(coll, 3);
        getCount(coll, &count);
        for (unsigned int i = 0U; i < count; i = i + 1U) {
            void* dev = (void*)0;
            WCollItem_ item = (WCollItem_)w_method_(coll, 4);
            if (item(coll, i, &dev) < 0 || dev == (void*)0) continue;
            char id[AUDIO_DEVICE_KEY_MAX];
            id[0] = (char)0;
            unsigned short* wid = (unsigned short*)0;
            WGetId_ getId = (WGetId_)w_method_(dev, 5);
            if (getId(dev, &wid) >= 0 && wid != (unsigned short*)0) {
                w_to_utf8_((const unsigned short*)wid, &id[0], (unsigned long)AUDIO_DEVICE_KEY_MAX);
                CoTaskMemFree((void*)wid);
            }
            struct AudioDeviceInfo info = audio_device_info_blank_();
            void* props = (void*)0;
            WOpenProps_ openProps = (WOpenProps_)w_method_(dev, 4);
            if (openProps(dev, W_STGM_READ, &props) >= 0 && props != (void*)0) {
                struct WPropKey_ key = w_pkey_friendly_name_();
                struct WPropVariant_ pv;
                safe_memset_raw_((void*)&pv, 0, sizeof(struct WPropVariant_));
                WGetValue_ getValue = (WGetValue_)w_method_(props, 5);
                if (getValue(props, &key, &pv) >= 0 && (int)pv.vt == W_VT_LPWSTR)
                    w_to_utf8_((const unsigned short*)pv.p, &info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX);
                PropVariantClear((void*)&pv);
                w_release_(props);
            }
            if (info.name[0] == (char)0) audio_strcpy_(&info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, "WASAPI endpoint");
            void* client = (void*)0;
            if (w_audio_client_(dev, &client) >= 0 && client != (void*)0) {
                WFormat_* mix = (WFormat_*)0;
                WGetMixFormat_ getMix = (WGetMixFormat_)w_method_(client, 8);
                if (getMix(client, (void**)&mix) >= 0 && mix != (WFormat_*)0) {
                    int ch = (int)mix->channels;
                    if (flow == W_E_RENDER) info.output_channels = ch; else info.input_channels = ch;
                    info.preferred_sample_rate = mix->rate;
                    audio_device_add_rate_(&info, mix->rate);
                    info.native_formats = w_format_of_((const WFormat_*)mix);
                    // Exclusive mode: which standard rates the device takes.
                    WIsSupported_ isSup = (WIsSupported_)w_method_(client, 7);
                    for (int r = 0; r < audio_standard_rate_count_(); r = r + 1) {
                        unsigned int rate = audio_standard_rate_(r);
                        struct WFormat_ f32 = w_make_format_(AUDIO_FORMAT_F32, ch, rate);
                        struct WFormat_ i16 = w_make_format_(AUDIO_FORMAT_I16, ch, rate);
                        if (isSup(client, W_SHAREMODE_EXCLUSIVE, &f32, (void**)0) == 0) {
                            audio_device_add_rate_(&info, rate);
                            info.native_formats = info.native_formats | AUDIO_FORMAT_F32;
                        } else if (isSup(client, W_SHAREMODE_EXCLUSIVE, &i16, (void**)0) == 0) {
                            audio_device_add_rate_(&info, rate);
                            info.native_formats = info.native_formats | AUDIO_FORMAT_I16;
                        }
                    }
                    CoTaskMemFree((void*)mix);
                }
                w_release_(client);
            }
            w_release_(dev);
            if (info.output_channels == 0 && info.input_channels == 0) continue;
            int isDefault = id[0] != (char)0 && audio_streq_(&id[0], defId);
            info.is_default_output = isDefault && flow == W_E_RENDER;
            info.is_default_input = isDefault && flow == W_E_CAPTURE;
            match (audio_host_add_(h, &info, &id[0], (long long)flow)) {
                case some(rec): {
                    if (rec.info.is_default_output) h.default_out = rec.info.id;
                    if (rec.info.is_default_input) h.default_in = rec.info.id;
                }
                case null: {}
            }
        }
        w_release_(coll);
    }
}

static void w_default_id_(void* en, int flow, &char out) {
    unsafe {
        out[0] = (char)0;
        void* dev = (void*)0;
        WGetDefault_ getDefault = (WGetDefault_)w_method_(en, 4);
        if (getDefault(en, flow, W_E_CONSOLE, &dev) < 0 || dev == (void*)0) return;
        unsigned short* wid = (unsigned short*)0;
        WGetId_ getId = (WGetId_)w_method_(dev, 5);
        if (getId(dev, &wid) >= 0 && wid != (unsigned short*)0) {
            w_to_utf8_((const unsigned short*)wid, out, (unsigned long)AUDIO_DEVICE_KEY_MAX);
            CoTaskMemFree((void*)wid);
        }
        w_release_(dev);
    }
}

static int audio_wasapi_probe_(&AudioHost_ h) {
    if (!w_coinit_()) return audio_fail_(&h.error, AUDIO_ERR_SYSTEM, "wasapi: CoInitializeEx failed");
    int hr = 0;
    void* en = w_enumerator_(&hr);
    if (en == (void*)0) return audio_fail_hex_(&h.error, AUDIO_ERR_UNAVAILABLE, "wasapi: no MMDeviceEnumerator", (unsigned int)hr);
    char defOut[AUDIO_DEVICE_KEY_MAX];
    char defIn[AUDIO_DEVICE_KEY_MAX];
    w_default_id_(en, W_E_RENDER, &defOut[0]);
    w_default_id_(en, W_E_CAPTURE, &defIn[0]);
    w_probe_flow_(h, en, W_E_RENDER, &defOut[0]);
    w_probe_flow_(h, en, W_E_CAPTURE, &defIn[0]);
    w_release_(en);
    return AUDIO_OK;
}

static void audio_wasapi_host_close_(&AudioHost_ h) { }

// ── streams ──────────────────────────────────────────────────────────────

struct WStream_ {
    void* client[2];          // IAudioClient per direction
    void* service[2];         // IAudioRenderClient / IAudioCaptureClient
    void* event[2];
    unsigned int buffer[2];   // endpoint buffer frames
    int exclusive;
    // capture FIFO (device format frames), filled by the stream thread only
    unsigned char* fifo;
    unsigned long fifo_frames;
    unsigned long fifo_read;
    unsigned long fifo_count;
    struct AudioStreamState_* st;
};

static ?&WStream_ w_of_(&AudioStreamState_ st) {
    unsafe { return (?&WStream_)st.backend; }
}

static void w_stream_release_(&WStream_ w) {
    for (int d = 0; d < 2; d = d + 1) {
        w_release_(w.service[d]);
        w_release_(w.client[d]);
        unsafe { if (w.event[d] != (void*)0) CloseHandle(w.event[d]); }
        w.service[d] = (void*)0;
        w.client[d] = (void*)0;
        w.event[d] = (void*)0;
    }
    unsafe { dealloc_raw_((void*)w.fifo); }
    w.fifo = (unsigned char*)0;
}

static long long w_hns_(unsigned int frames, unsigned int rate) {
    return (long long)((10000000.0 * (double)frames) / (double)rate + 0.5);
}

// Opens the endpoint for one direction: format, Initialize, event, service.
static int w_open_dir_(&AudioStreamState_ st, &WStream_ w, void* en, int d) {
    char* key = (char*)0;
    unsafe { key = (char*)&st.device_key[d][0]; }
    unsigned short wid[AUDIO_DEVICE_KEY_MAX];
    void* dev = (void*)0;
    int hr = 0;
    unsafe {
        if (MultiByteToWideChar(W_CP_UTF8, 0U, (const char*)key, -1, &wid[0], (int)AUDIO_DEVICE_KEY_MAX) <= 0)
            return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "wasapi: bad endpoint id");
        WGetDevice_ getDevice = (WGetDevice_)w_method_(en, 5);
        hr = getDevice(en, (const unsigned short*)&wid[0], &dev);
        if (hr < 0 || dev == (void*)0) return audio_fail_hex_(&st.error, AUDIO_ERR_INVALID_DEVICE, "wasapi: endpoint gone", (unsigned int)hr);
    }
    void* client = (void*)0;
    hr = w_audio_client_(dev, &client);
    if (hr < 0) {
        w_release_(dev);
        return audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: IMMDevice::Activate", (unsigned int)hr);
    }
    int rc = AUDIO_OK;
    unsafe {
        WFormat_* mix = (WFormat_*)0;
        WGetMixFormat_ getMix = (WGetMixFormat_)w_method_(client, 8);
        hr = getMix(client, (void**)&mix);
        if (hr < 0 || mix == (WFormat_*)0) {
            w_release_(client); w_release_(dev);
            return audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: GetMixFormat", (unsigned int)hr);
        }
        int channels = (int)mix->channels;
        if (st.user_channels[d] + st.first_channel[d] > channels) {
            CoTaskMemFree((void*)mix); w_release_(client); w_release_(dev);
            return audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "wasapi: the endpoint has too few channels");
        }
        long long defPeriod = 0LL;
        long long minPeriod = 0LL;
        WGetPeriod_ getPeriod = (WGetPeriod_)w_method_(client, 9);
        getPeriod(client, &defPeriod, &minPeriod);
        // Our period: the request, else the engine's.
        unsigned int frames = st.buffer_frames;
        if (frames == 0U) frames = (unsigned int)((defPeriod * (long long)st.sample_rate + 5000000LL) / 10000000LL);
        if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U)
            frames = (unsigned int)((minPeriod * (long long)st.sample_rate + 9999999LL) / 10000000LL);
        if (frames == 0U) frames = 480U;
        struct WFormat_ fmt;
        WInitialize_ init = (WInitialize_)w_method_(client, 3);
        if (w.exclusive) {
            WIsSupported_ isSup = (WIsSupported_)w_method_(client, 7);
            unsigned int order[5];
            order[0] = st.user_format; order[1] = AUDIO_FORMAT_F32; order[2] = AUDIO_FORMAT_I32;
            order[3] = AUDIO_FORMAT_I24; order[4] = AUDIO_FORMAT_I16;
            unsigned int chosen = 0U;
            for (int i = 0; i < 5 && chosen == 0U; i = i + 1) {
                fmt = w_make_format_(order[i], channels, st.sample_rate);
                if (isSup(client, W_SHAREMODE_EXCLUSIVE, &fmt, (void**)0) == 0) chosen = order[i];
            }
            if (chosen == 0U) rc = audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "wasapi: no exclusive-mode format at this rate");
            else {
                long long period = w_hns_(frames, st.sample_rate);
                if (period < minPeriod) period = minPeriod;
                hr = init(client, W_SHAREMODE_EXCLUSIVE, W_FLAG_EVENTCALLBACK, period, period, &fmt, (const WGuid_*)0);
                if ((unsigned int)hr == W_E_NOT_ALIGNED) {
                    // Re-align to the size the driver wants, on a fresh client.
                    unsigned int aligned = 0U;
                    WGetBufferSize_ getSize = (WGetBufferSize_)w_method_(client, 4);
                    getSize(client, &aligned);
                    w_release_(client);
                    client = (void*)0;
                    period = w_hns_(aligned, st.sample_rate);
                    hr = w_audio_client_(dev, &client);
                    if (hr >= 0) {
                        init = (WInitialize_)w_method_(client, 3);
                        hr = init(client, W_SHAREMODE_EXCLUSIVE, W_FLAG_EVENTCALLBACK, period, period, &fmt, (const WGuid_*)0);
                    }
                }
                if (hr < 0) {
                    int code = (unsigned int)hr == W_E_DEVICE_IN_USE ? AUDIO_ERR_DEVICE_BUSY : AUDIO_ERR_SYSTEM;
                    rc = audio_fail_hex_(&st.error, code, "wasapi: exclusive IAudioClient::Initialize", (unsigned int)hr);
                }
                st.device_format[d] = chosen;
            }
        } else {
            // Shared: the mix format; the engine resamples another rate.
            unsigned long fmtBytes = 18UL + (unsigned long)mix->cb_size;
            if (fmtBytes > sizeof(struct WFormat_)) fmtBytes = sizeof(struct WFormat_);
            safe_memset_raw_((void*)&fmt, 0, sizeof(struct WFormat_));
            safe_memcpy_raw_((void*)&fmt, (const void*)mix, fmtBytes);
            unsigned int flags = W_FLAG_EVENTCALLBACK;
            if (fmt.rate != st.sample_rate) {
                fmt.rate = st.sample_rate;
                fmt.avg_bytes = st.sample_rate * (unsigned int)fmt.block_align;
                flags = flags | W_FLAG_AUTOCONVERTPCM | W_FLAG_SRC_DEFAULT_QUALITY;
            }
            unsigned int periods = st.buffer_count < 2U ? 3U : st.buffer_count + 1U;
            long long dur = w_hns_(frames * periods, st.sample_rate);
            if (dur < defPeriod * 2LL) dur = defPeriod * 2LL;
            hr = init(client, W_SHAREMODE_SHARED, flags, dur, 0LL, &fmt, (const WGuid_*)0);
            if (hr < 0) rc = audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: IAudioClient::Initialize", (unsigned int)hr);
            st.device_format[d] = w_format_of_(&fmt);
            if (rc == AUDIO_OK && st.device_format[d] == 0U)
                rc = audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "wasapi: unsupported mix format");
        }
        CoTaskMemFree((void*)mix);
        w_release_(dev);
        if (rc != AUDIO_OK) { w_release_(client); return rc; }
        unsigned int bufFrames = 0U;
        WGetBufferSize_ getSize = (WGetBufferSize_)w_method_(client, 4);
        getSize(client, &bufFrames);
        if (w.exclusive) frames = bufFrames;          // one period per event
        else if (frames > bufFrames / 2U) frames = bufFrames / 2U;
        w.buffer[d] = bufFrames;
        w.client[d] = client;
        w.event[d] = CreateEventA((void*)0, 0, 0, (const char*)0);
        if (w.event[d] == (void*)0) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: CreateEvent");
        WSetEvent_ setEvent = (WSetEvent_)w_method_(client, 13);
        hr = setEvent(client, w.event[d]);
        if (hr < 0) return audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: SetEventHandle", (unsigned int)hr);
        struct WGuid_ iid = d == AUDIO_DIR_OUT ? w_iid_render_client_() : w_iid_capture_client_();
        WGetService_ getService = (WGetService_)w_method_(client, 14);
        hr = getService(client, &iid, &w.service[d]);
        if (hr < 0) return audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: GetService", (unsigned int)hr);
        long long lat = 0LL;
        WGetLatency_ getLatency = (WGetLatency_)w_method_(client, 5);
        getLatency(client, &lat);
        st.latency[d] = (long)((lat * (long long)st.sample_rate) / 10000000LL) + (long)bufFrames;
        st.device_channels[d] = channels;
        st.device_interleaved[d] = 1;
        // Both directions run on the output's period in duplex.
        if (d == AUDIO_DIR_OUT || !st.mode_out) st.buffer_frames = frames;
    }
    return AUDIO_OK;
}

static int audio_wasapi_open_(&AudioHost_ h, &AudioStreamState_ st) {
    if (st.device_name[0] != (char)0)
        return audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "wasapi: device_name is not used; pass a device id");
    if (st.mode_out && st.device_native[AUDIO_DIR_OUT] != (long long)W_E_RENDER)
        return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "wasapi: output needs a render endpoint");
    if (st.mode_in && st.device_native[AUDIO_DIR_IN] != (long long)W_E_CAPTURE)
        return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "wasapi: input needs a capture endpoint");
    if (!w_coinit_()) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: CoInitializeEx failed");
    int hr = 0;
    void* en = w_enumerator_(&hr);
    if (en == (void*)0) return audio_fail_hex_(&st.error, AUDIO_ERR_UNAVAILABLE, "wasapi: no MMDeviceEnumerator", (unsigned int)hr);
    ?&WStream_ wp = null;
    unsafe { wp = (?&WStream_)alloc_zeroed_raw_(sizeof(struct WStream_)); }
    match (wp) {
        case null: {
            w_release_(en);
            return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "wasapi: out of memory");
        }
        case some(w): {
            unsafe { w.st = (struct AudioStreamState_*)st; }
            w.exclusive = (st.flags & AUDIO_FLAG_EXCLUSIVE) != 0U;
            int rc = AUDIO_OK;
            if (st.mode_out) rc = w_open_dir_(st, w, en, AUDIO_DIR_OUT);
            if (rc == AUDIO_OK && st.mode_in) rc = w_open_dir_(st, w, en, AUDIO_DIR_IN);
            w_release_(en);
            if (rc == AUDIO_OK && st.mode_in) {
                // Capture FIFO: room for several endpoint buffers.
                w.fifo_frames = (unsigned long)w.buffer[AUDIO_DIR_IN] * 4UL + (unsigned long)st.buffer_frames * 2UL;
                unsigned long fb = (unsigned long)st.device_channels[AUDIO_DIR_IN] *
                                   (unsigned long)audio_format_bytes(st.device_format[AUDIO_DIR_IN]);
                unsafe { w.fifo = (unsigned char*)alloc_zeroed_raw_(w.fifo_frames * fb); }
                if (w.fifo == (unsigned char*)0) rc = audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "wasapi: out of memory");
            }
            if (rc != AUDIO_OK) {
                w_stream_release_(w);
                unsafe { dealloc_raw_((void*)w); }
                return rc;
            }
            st.max_frames = st.buffer_frames;
            if (st.buffer_count == 0U) st.buffer_count = 2U;
            unsafe { st.backend = (void*)w; }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

// Moves every pending capture packet into the FIFO.
static void w_capture_drain_(&AudioStreamState_ st, &WStream_ w) {
    unsigned long fb = (unsigned long)st.device_channels[AUDIO_DIR_IN] *
                       (unsigned long)audio_format_bytes(st.device_format[AUDIO_DIR_IN]);
    unsafe {
        void* cap = w.service[AUDIO_DIR_IN];
        WCaptureNext_ next = (WCaptureNext_)w_method_(cap, 5);
        WCaptureGet_ get = (WCaptureGet_)w_method_(cap, 3);
        WCaptureRelease_ rel = (WCaptureRelease_)w_method_(cap, 4);
        while (1) {
            unsigned int packet = 0U;
            if (next(cap, &packet) < 0 || packet == 0U) break;
            unsigned char* data = (unsigned char*)0;
            unsigned int frames = 0U;
            unsigned int flags = 0U;
            if (get(cap, &data, &frames, &flags, (unsigned long long*)0, (unsigned long long*)0) < 0) break;
            if ((flags & W_BUFFERFLAGS_DISCONTINUITY) != 0U) audio_stream_flag_status_(st, AUDIO_STATUS_INPUT_OVERFLOW);
            for (unsigned int f = 0U; f < frames; f = f + 1U) {
                if (w.fifo_count >= w.fifo_frames) {      // full: the oldest frame goes
                    w.fifo_read = (w.fifo_read + 1UL) % w.fifo_frames;
                    w.fifo_count = w.fifo_count - 1UL;
                    audio_stream_flag_status_(st, AUDIO_STATUS_INPUT_OVERFLOW);
                }
                unsigned long at = ((w.fifo_read + w.fifo_count) % w.fifo_frames) * fb;
                for (unsigned long b = 0UL; b < fb; b = b + 1UL)
                    w.fifo[at + b] = (flags & W_BUFFERFLAGS_SILENT) != 0U ? (unsigned char)0 : data[(unsigned long)f * fb + b];
                w.fifo_count = w.fifo_count + 1UL;
            }
            rel(cap, frames);
        }
    }
}

// One period of input from the FIFO into device_buf[IN] (silence if short).
static void w_capture_take_(&AudioStreamState_ st, &WStream_ w, unsigned int frames) {
    unsigned long fb = (unsigned long)st.device_channels[AUDIO_DIR_IN] *
                       (unsigned long)audio_format_bytes(st.device_format[AUDIO_DIR_IN]);
    unsafe {
        unsigned char* dst = (unsigned char*)st.device_buf[AUDIO_DIR_IN];
        if (w.fifo_count < (unsigned long)frames) audio_stream_flag_status_(st, AUDIO_STATUS_INPUT_OVERFLOW);
        for (unsigned long f = 0UL; f < (unsigned long)frames; f = f + 1UL) {
            if (w.fifo_count > 0UL) {
                unsigned long at = w.fifo_read * fb;
                for (unsigned long b = 0UL; b < fb; b = b + 1UL) dst[f * fb + b] = w.fifo[at + b];
                w.fifo_read = (w.fifo_read + 1UL) % w.fifo_frames;
                w.fifo_count = w.fifo_count - 1UL;
            } else {
                for (unsigned long b = 0UL; b < fb; b = b + 1UL) dst[f * fb + b] = (unsigned char)0;
            }
        }
    }
}

// Writes device_buf[OUT] (frames) into the render buffer. <0 on failure.
static int w_render_put_(&AudioStreamState_ st, &WStream_ w, unsigned int frames, int silent) {
    unsigned long bytes = (unsigned long)frames * audio_stream_device_frame_bytes_(st, AUDIO_DIR_OUT);
    unsafe {
        void* rc = w.service[AUDIO_DIR_OUT];
        WRenderGet_ get = (WRenderGet_)w_method_(rc, 3);
        WRenderRelease_ rel = (WRenderRelease_)w_method_(rc, 4);
        unsigned char* data = (unsigned char*)0;
        if (get(rc, frames, &data) < 0 || data == (unsigned char*)0) return -1;
        if (!silent) {
            const unsigned char* src = (const unsigned char*)st.device_buf[AUDIO_DIR_OUT];
            for (unsigned long b = 0UL; b < bytes; b = b + 1UL) data[b] = src[b];
        }
        rel(rc, frames, silent ? W_BUFFERFLAGS_SILENT : 0U);
    }
    return 0;
}

static unsigned int w_padding_(&WStream_ w) {
    unsigned int pad = 0U;
    unsafe {
        WGetBufferSize_ getPadding = (WGetBufferSize_)w_method_(w.client[AUDIO_DIR_OUT], 6);
        getPadding(w.client[AUDIO_DIR_OUT], &pad);
    }
    return pad;
}

static void w_client_call_(void* client, int index) {
    unsafe {
        if (client == (void*)0) return;
        WClientNoArg_ f = (WClientNoArg_)w_method_(client, index);
        f(client);
    }
}

// One callback period (input from the FIFO, output to the endpoint).
static int w_period_(&AudioStreamState_ st, &WStream_ w) {
    unsigned int frames = st.buffer_frames;
    if (st.mode_in) w_capture_take_(st, w, frames);
    int r = audio_stream_process_(st, frames);
    if (st.mode_out && w_render_put_(st, w, frames, 0) < 0) {
        audio_stream_set_rt_error_(st, AUDIO_ERR_SYSTEM, "wasapi: IAudioRenderClient::GetBuffer failed");
        return AUDIO_CB_ABORT;
    }
    return r;
}

static void w_thread_(&AudioStreamState_ st) {
    w_coinit_();
    audio_thread_make_realtime_(st.priority);      // MMCSS "Pro Audio"
    match (w_of_(st)) {
        case some(w): {
            int how = AUDIO_CB_CONTINUE;
            void* wait = st.mode_out ? w.event[AUDIO_DIR_OUT] : w.event[AUDIO_DIR_IN];
            int primed = 0;
            while (1) {
                how = audio_stream_stop_requested_(st);
                if (how != AUDIO_CB_CONTINUE) break;
                unsigned int wr = 0U;
                unsafe { wr = WaitForSingleObject(wait, 2000U); }
                if (wr == W_WAIT_TIMEOUT) {
                    audio_stream_set_rt_error_(st, AUDIO_ERR_SYSTEM, "wasapi: the endpoint stopped signalling");
                    how = AUDIO_CB_ABORT;
                    break;
                }
                if (st.mode_in) w_capture_drain_(st, w);
                int r = AUDIO_CB_CONTINUE;
                if (st.mode_out) {
                    if (w.exclusive) {
                        r = w_period_(st, w);
                    } else {
                        unsigned int pad = w_padding_(w);
                        if (pad == 0U && primed) audio_stream_flag_status_(st, AUDIO_STATUS_OUTPUT_UNDERFLOW);
                        primed = 1;
                        unsigned int avail = w.buffer[AUDIO_DIR_OUT] - pad;
                        while (avail >= st.buffer_frames && r == AUDIO_CB_CONTINUE) {
                            r = w_period_(st, w);
                            avail = avail - st.buffer_frames;
                        }
                    }
                } else {
                    while (w.fifo_count >= (unsigned long)st.buffer_frames && r == AUDIO_CB_CONTINUE) r = w_period_(st, w);
                }
                if (r != AUDIO_CB_CONTINUE) { how = r; break; }
            }
            if (st.mode_out && how == AUDIO_CB_STOP) {
                // Drain: wait until the endpoint has played what's queued.
                for (int t = 0; t < 400 && w_padding_(w) > 0U; t = t + 1) audio_sleep_us_(1000LL);
            }
            for (int d = 0; d < 2; d = d + 1) {
                w_client_call_(w.client[d], 11);           // Stop
                if (how != AUDIO_CB_STOP) w_client_call_(w.client[d], 12);   // Reset
            }
        }
        case null: {}
    }
    audio_stream_mark_stopped_(st, null);
}

static int audio_wasapi_start_(&AudioStreamState_ st) {
    match (w_of_(st)) {
        case some(w): {
            w.fifo_read = 0UL;
            w.fifo_count = 0UL;
            if (st.mode_out) {
                // Pre-fill the render buffer with silence so the first event
                // doesn't find it empty.
                for (int d = 0; d < 2; d = d + 1) w_client_call_(w.client[d], 12);   // Reset
                unsigned int fill = w.exclusive ? w.buffer[AUDIO_DIR_OUT] : w.buffer[AUDIO_DIR_OUT] - w_padding_(w);
                if (fill > 0U && w_render_put_(st, w, fill, 1) < 0)
                    return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: could not pre-fill the render buffer");
            }
            int rc = audio_stream_thread_start_(st, w_thread_);
            if (rc != AUDIO_OK) return rc;
            unsafe {
                for (int d = 0; d < 2; d = d + 1) {
                    if (w.client[d] == (void*)0) continue;
                    WClientNoArg_ start = (WClientNoArg_)w_method_(w.client[d], 10);
                    int hr = start(w.client[d]);
                    if (hr < 0) {
                        audio_stream_request_stop_(st, AUDIO_CB_ABORT);
                        audio_stream_thread_join_(st);
                        return audio_fail_hex_(&st.error, AUDIO_ERR_SYSTEM, "wasapi: IAudioClient::Start", (unsigned int)hr);
                    }
                }
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "wasapi: stream not open");
}

static int audio_wasapi_stop_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    return AUDIO_OK;
}

static void audio_wasapi_close_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    match (w_of_(st)) {
        case some(w): {
            w_stream_release_(w);
            unsafe { dealloc_raw_((void*)w); }
        }
        case null: {}
    }
    unsafe { st.backend = (void*)0; }
}

static struct AudioBackendVt_ g_audio_wasapi_vt_;
static int g_audio_wasapi_vt_ready_ = 0;

&static AudioBackendVt_ audio_wasapi_backend_() {
    if (!g_audio_wasapi_vt_ready_) {
        g_audio_wasapi_vt_.api = AUDIO_API_WASAPI;
        g_audio_wasapi_vt_.available = audio_wasapi_available_;
        g_audio_wasapi_vt_.probe = audio_wasapi_probe_;
        g_audio_wasapi_vt_.host_close = audio_wasapi_host_close_;
        g_audio_wasapi_vt_.open = audio_wasapi_open_;
        g_audio_wasapi_vt_.start = audio_wasapi_start_;
        g_audio_wasapi_vt_.stop = audio_wasapi_stop_;
        g_audio_wasapi_vt_.abort = audio_wasapi_stop_;
        g_audio_wasapi_vt_.close = audio_wasapi_close_;
        g_audio_wasapi_vt_ready_ = 1;
    }
    return &g_audio_wasapi_vt_;
}

} // namespace std
