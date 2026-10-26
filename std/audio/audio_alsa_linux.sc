// SafeC Standard Library — audio backend: ALSA PCM (Linux). See audio.h.
//
// libasound.so.2 is loaded with dlopen() on first use, so a program built
// with std/audio starts (and probes zero ALSA devices) on a system without
// it. Devices: ALSA's "default" PCM when it opens, then every hw:CARD,DEV
// PCM the control interface lists — probed for channels, formats and the
// standard rates in non-blocking mode, so a busy device doesn't stall the
// probe. AudioStreamOptions.device_name opens any other PCM by name
// ("null", "plughw:1,0", a ~/.asoundrc plugin) — handy without hardware.
//
// A stream runs one thread std starts: blocking snd_pcm_readi()/writei()
// (or readn()/writen() when the device only does non-interleaved access)
// of one period, the callback in between. xruns (-EPIPE) are recovered
// with snd_pcm_prepare() and reported as AUDIO_STATUS_INPUT_OVERFLOW /
// _OUTPUT_UNDERFLOW in the next callback. The device format is the
// stream's when the hardware takes it, else the best one it does (f32,
// i32, packed i24, i16, f64, i8) with std converting. A hw: device that
// refuses the rate is reopened through ALSA's plug layer ("plughw:"),
// unless AUDIO_FLAG_EXCLUSIVE asked for the raw device. ALSA's own error
// printing is silenced; its messages land in AudioError instead.
#pragma once
#include <std/stddef.h>
#include <std/audio/audio_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>

#define ALSA_STREAM_PLAYBACK 0
#define ALSA_STREAM_CAPTURE  1
#define ALSA_NONBLOCK        1
#define ALSA_ACCESS_RW_INTERLEAVED    3
#define ALSA_ACCESS_RW_NONINTERLEAVED 4
#define ALSA_FMT_S8         0
#define ALSA_FMT_S16_LE     2
#define ALSA_FMT_S32_LE     10
#define ALSA_FMT_FLOAT_LE   14
#define ALSA_FMT_FLOAT64_LE 16
#define ALSA_FMT_S24_3LE    32
#define ALSA_EPIPE   32
#define ALSA_ESTRPIPE 86
#define ALSA_EAGAIN  11
#define ALSA_EBUSY   16

namespace std {

// ── libasound, loaded at run time ────────────────────────────────────────
// libasound's handles and parameter blocks (opaque; the blocks are
// allocated at the size snd_*_sizeof() reports).
struct SndPcm_         { char opaque_; };   // snd_pcm_t
struct SndHwParams_    { char opaque_; };   // snd_pcm_hw_params_t
struct SndSwParams_    { char opaque_; };   // snd_pcm_sw_params_t
struct SndCtl_         { char opaque_; };   // snd_ctl_t
struct SndCtlCardInfo_ { char opaque_; };   // snd_ctl_card_info_t
struct SndPcmInfo_     { char opaque_; };   // snd_pcm_info_t

// snd_lib_error_handler_t (variadic in C; see audio_alsa_quiet_).
typedef fn void(const char*, int, const char*, int, const char*) AlsaErrorHandler_;

typedef fn int(SndPcm_**, const char*, int, int) AlsaPcmOpen_;
typedef fn int(SndPcm_*) AlsaPcm1_;                                  // close/prepare/start/drop/drain
typedef fn int(SndPcm_*, int) AlsaPcmNonblock_;
typedef fn size_t() AlsaSizeof_;
typedef fn int(SndPcm_*, SndHwParams_*) AlsaHwAny_;                  // hw_params_any, hw_params
typedef fn int(SndPcm_*, SndSwParams_*) AlsaSwAny_;                  // sw_params_current, sw_params
typedef fn int(SndPcm_*, SndHwParams_*, int) AlsaHwSetInt_;          // set_access, test/set_format
typedef fn int(SndPcm_*, SndHwParams_*, unsigned int) AlsaHwSetUint_;   // set_channels, set_rate_resample
typedef fn int(const SndHwParams_*, unsigned int*) AlsaHwGetUint_;   // get_channels_max/min
typedef fn int(SndPcm_*, SndHwParams_*, unsigned int, int) AlsaHwTestRate_;
typedef fn int(SndPcm_*, SndHwParams_*, unsigned int*, int*) AlsaHwSetUintNear_;   // set_rate_near, set_periods_near
typedef fn int(SndPcm_*, SndHwParams_*, size_t*, int*) AlsaHwSetFramesNear_;      // set_period_size_near
typedef fn int(SndPcm_*, SndHwParams_*, size_t*) AlsaHwSetBufferNear_;            // set_buffer_size_near
typedef fn int(const SndHwParams_*, size_t*, int*) AlsaHwGetPeriod_;
typedef fn int(const SndHwParams_*, size_t*) AlsaHwGetBuffer_;
typedef fn int(const SndHwParams_*, unsigned int*, int*) AlsaHwGetRate_;
typedef fn int(SndPcm_*, SndSwParams_*, size_t) AlsaSwSetFrames_;    // set_start_threshold, set_avail_min
typedef fn ssize_t(SndPcm_*, const unsigned char*, size_t) AlsaWritei_;
typedef fn ssize_t(SndPcm_*, unsigned char*, size_t) AlsaReadi_;
typedef fn ssize_t(SndPcm_*, unsigned char**, size_t) AlsaXfern_;    // one pointer per channel plane
typedef fn int(SndPcm_*, int, int) AlsaRecover_;
typedef fn int(SndPcm_*, SndPcm_*) AlsaLink_;
typedef fn const char*(int) AlsaStrerror_;
typedef fn int(AlsaErrorHandler_) AlsaSetHandler_;
typedef fn int(int*) AlsaCardNext_;
typedef fn int(SndCtl_**, const char*, int) AlsaCtlOpen_;
typedef fn int(SndCtl_*) AlsaCtlClose_;
typedef fn int(SndCtl_*, SndCtlCardInfo_*) AlsaCtlCardInfo_;
typedef fn int(SndCtl_*, SndPcmInfo_*) AlsaCtlPcmInfo_;
typedef fn const char*(const SndCtlCardInfo_*) AlsaCardGetName_;
typedef fn const char*(const SndPcmInfo_*) AlsaPcmGetName_;
typedef fn int(SndCtl_*, int*) AlsaCtlNextDev_;
typedef fn void(SndPcmInfo_*, unsigned int) AlsaInfoSetUint_;        // pcm_info_set_device/subdevice
typedef fn void(SndPcmInfo_*, int) AlsaInfoSetStream_;

struct AlsaLib_ {
    int loaded;      // 0 not tried, 1 ok, -1 unavailable
    AlsaPcmOpen_ pcm_open;
    AlsaPcm1_ pcm_close;
    AlsaPcm1_ pcm_prepare;
    AlsaPcm1_ pcm_start;
    AlsaPcm1_ pcm_drop;
    AlsaPcm1_ pcm_drain;
    AlsaPcmNonblock_ pcm_nonblock;
    AlsaSizeof_ hw_sizeof;
    AlsaSizeof_ sw_sizeof;
    AlsaSizeof_ ctl_card_info_sizeof;
    AlsaSizeof_ pcm_info_sizeof;
    AlsaHwAny_ hw_any;
    AlsaHwAny_ hw_params;
    AlsaSwAny_ sw_current;
    AlsaSwAny_ sw_params;
    AlsaHwSetInt_ hw_set_access;
    AlsaHwSetInt_ hw_test_access;
    AlsaHwSetInt_ hw_test_format;
    AlsaHwSetInt_ hw_set_format;
    AlsaHwSetUint_ hw_set_channels;
    AlsaHwSetUint_ hw_set_rate_resample;
    AlsaHwGetUint_ hw_get_channels_max;
    AlsaHwGetUint_ hw_get_channels_min;
    AlsaHwTestRate_ hw_test_rate;
    AlsaHwSetUintNear_ hw_set_rate_near;
    AlsaHwSetUintNear_ hw_set_periods_near;
    AlsaHwSetFramesNear_ hw_set_period_size_near;
    AlsaHwSetBufferNear_ hw_set_buffer_size_near;
    AlsaHwGetPeriod_ hw_get_period_size;
    AlsaHwGetBuffer_ hw_get_buffer_size;
    AlsaHwGetRate_ hw_get_rate;
    AlsaSwSetFrames_ sw_set_start_threshold;
    AlsaSwSetFrames_ sw_set_avail_min;
    AlsaWritei_ writei;
    AlsaReadi_ readi;
    AlsaXfern_ writen;
    AlsaXfern_ readn;
    AlsaRecover_ recover;
    AlsaLink_ link;
    AlsaStrerror_ strerror;
    AlsaSetHandler_ set_error_handler;
    AlsaCardNext_ card_next;
    AlsaCtlOpen_ ctl_open;
    AlsaCtlClose_ ctl_close;
    AlsaCtlCardInfo_ ctl_card_info;
    AlsaCardGetName_ card_info_get_name;
    AlsaCtlNextDev_ ctl_pcm_next_device;
    AlsaCtlPcmInfo_ ctl_pcm_info;
    AlsaInfoSetUint_ pcm_info_set_device;
    AlsaInfoSetUint_ pcm_info_set_subdevice;
    AlsaInfoSetStream_ pcm_info_set_stream;
    AlsaPcmGetName_ pcm_info_get_name;
};

static struct AlsaLib_ g_alsa_;

// ALSA prints its errors to stderr by default; std reports them through
// AudioError. (The handler is variadic in C; the extra arguments are
// ignored, which every Linux ABI allows.)
static void audio_alsa_quiet_(const char* file, int line, const char* function, int err, const char* fmt) { }

static fn void() audio_alsa_sym_(?&AudioLib_ lib, const &char name, &int ok) {
    return audio_dlsym_or_(lib, name, ok);
}

static int audio_alsa_load_() {
    if (g_alsa_.loaded != 0) return g_alsa_.loaded > 0;
    g_alsa_.loaded = -1;
    ?&AudioLib_ lib = audio_dlopen2_("libasound.so.2", "libasound.so");
    if (lib.is_null()) return 0;
    int ok = 1;
    unsafe {
        g_alsa_.pcm_open = (AlsaPcmOpen_)audio_alsa_sym_(lib, "snd_pcm_open", &ok);
        g_alsa_.pcm_close = (AlsaPcm1_)audio_alsa_sym_(lib, "snd_pcm_close", &ok);
        g_alsa_.pcm_prepare = (AlsaPcm1_)audio_alsa_sym_(lib, "snd_pcm_prepare", &ok);
        g_alsa_.pcm_start = (AlsaPcm1_)audio_alsa_sym_(lib, "snd_pcm_start", &ok);
        g_alsa_.pcm_drop = (AlsaPcm1_)audio_alsa_sym_(lib, "snd_pcm_drop", &ok);
        g_alsa_.pcm_drain = (AlsaPcm1_)audio_alsa_sym_(lib, "snd_pcm_drain", &ok);
        g_alsa_.pcm_nonblock = (AlsaPcmNonblock_)audio_alsa_sym_(lib, "snd_pcm_nonblock", &ok);
        g_alsa_.hw_sizeof = (AlsaSizeof_)audio_alsa_sym_(lib, "snd_pcm_hw_params_sizeof", &ok);
        g_alsa_.sw_sizeof = (AlsaSizeof_)audio_alsa_sym_(lib, "snd_pcm_sw_params_sizeof", &ok);
        g_alsa_.ctl_card_info_sizeof = (AlsaSizeof_)audio_alsa_sym_(lib, "snd_ctl_card_info_sizeof", &ok);
        g_alsa_.pcm_info_sizeof = (AlsaSizeof_)audio_alsa_sym_(lib, "snd_pcm_info_sizeof", &ok);
        g_alsa_.hw_any = (AlsaHwAny_)audio_alsa_sym_(lib, "snd_pcm_hw_params_any", &ok);
        g_alsa_.hw_params = (AlsaHwAny_)audio_alsa_sym_(lib, "snd_pcm_hw_params", &ok);
        g_alsa_.sw_current = (AlsaSwAny_)audio_alsa_sym_(lib, "snd_pcm_sw_params_current", &ok);
        g_alsa_.sw_params = (AlsaSwAny_)audio_alsa_sym_(lib, "snd_pcm_sw_params", &ok);
        g_alsa_.hw_set_access = (AlsaHwSetInt_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_access", &ok);
        g_alsa_.hw_test_access = (AlsaHwSetInt_)audio_alsa_sym_(lib, "snd_pcm_hw_params_test_access", &ok);
        g_alsa_.hw_test_format = (AlsaHwSetInt_)audio_alsa_sym_(lib, "snd_pcm_hw_params_test_format", &ok);
        g_alsa_.hw_set_format = (AlsaHwSetInt_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_format", &ok);
        g_alsa_.hw_set_channels = (AlsaHwSetUint_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_channels", &ok);
        g_alsa_.hw_set_rate_resample = (AlsaHwSetUint_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_rate_resample", &ok);
        g_alsa_.hw_get_channels_max = (AlsaHwGetUint_)audio_alsa_sym_(lib, "snd_pcm_hw_params_get_channels_max", &ok);
        g_alsa_.hw_get_channels_min = (AlsaHwGetUint_)audio_alsa_sym_(lib, "snd_pcm_hw_params_get_channels_min", &ok);
        g_alsa_.hw_test_rate = (AlsaHwTestRate_)audio_alsa_sym_(lib, "snd_pcm_hw_params_test_rate", &ok);
        g_alsa_.hw_set_rate_near = (AlsaHwSetUintNear_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_rate_near", &ok);
        g_alsa_.hw_set_periods_near = (AlsaHwSetUintNear_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_periods_near", &ok);
        g_alsa_.hw_set_period_size_near = (AlsaHwSetFramesNear_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_period_size_near", &ok);
        g_alsa_.hw_set_buffer_size_near = (AlsaHwSetBufferNear_)audio_alsa_sym_(lib, "snd_pcm_hw_params_set_buffer_size_near", &ok);
        g_alsa_.hw_get_period_size = (AlsaHwGetPeriod_)audio_alsa_sym_(lib, "snd_pcm_hw_params_get_period_size", &ok);
        g_alsa_.hw_get_buffer_size = (AlsaHwGetBuffer_)audio_alsa_sym_(lib, "snd_pcm_hw_params_get_buffer_size", &ok);
        g_alsa_.hw_get_rate = (AlsaHwGetRate_)audio_alsa_sym_(lib, "snd_pcm_hw_params_get_rate", &ok);
        g_alsa_.sw_set_start_threshold = (AlsaSwSetFrames_)audio_alsa_sym_(lib, "snd_pcm_sw_params_set_start_threshold", &ok);
        g_alsa_.sw_set_avail_min = (AlsaSwSetFrames_)audio_alsa_sym_(lib, "snd_pcm_sw_params_set_avail_min", &ok);
        g_alsa_.writei = (AlsaWritei_)audio_alsa_sym_(lib, "snd_pcm_writei", &ok);
        g_alsa_.readi = (AlsaReadi_)audio_alsa_sym_(lib, "snd_pcm_readi", &ok);
        g_alsa_.writen = (AlsaXfern_)audio_alsa_sym_(lib, "snd_pcm_writen", &ok);
        g_alsa_.readn = (AlsaXfern_)audio_alsa_sym_(lib, "snd_pcm_readn", &ok);
        g_alsa_.recover = (AlsaRecover_)audio_alsa_sym_(lib, "snd_pcm_recover", &ok);
        g_alsa_.link = (AlsaLink_)audio_alsa_sym_(lib, "snd_pcm_link", &ok);
        g_alsa_.strerror = (AlsaStrerror_)audio_alsa_sym_(lib, "snd_strerror", &ok);
        g_alsa_.set_error_handler = (AlsaSetHandler_)audio_alsa_sym_(lib, "snd_lib_error_set_handler", &ok);
        g_alsa_.card_next = (AlsaCardNext_)audio_alsa_sym_(lib, "snd_card_next", &ok);
        g_alsa_.ctl_open = (AlsaCtlOpen_)audio_alsa_sym_(lib, "snd_ctl_open", &ok);
        g_alsa_.ctl_close = (AlsaCtlClose_)audio_alsa_sym_(lib, "snd_ctl_close", &ok);
        g_alsa_.ctl_card_info = (AlsaCtlCardInfo_)audio_alsa_sym_(lib, "snd_ctl_card_info", &ok);
        g_alsa_.card_info_get_name = (AlsaCardGetName_)audio_alsa_sym_(lib, "snd_ctl_card_info_get_name", &ok);
        g_alsa_.ctl_pcm_next_device = (AlsaCtlNextDev_)audio_alsa_sym_(lib, "snd_ctl_pcm_next_device", &ok);
        g_alsa_.ctl_pcm_info = (AlsaCtlPcmInfo_)audio_alsa_sym_(lib, "snd_ctl_pcm_info", &ok);
        g_alsa_.pcm_info_set_device = (AlsaInfoSetUint_)audio_alsa_sym_(lib, "snd_pcm_info_set_device", &ok);
        g_alsa_.pcm_info_set_subdevice = (AlsaInfoSetUint_)audio_alsa_sym_(lib, "snd_pcm_info_set_subdevice", &ok);
        g_alsa_.pcm_info_set_stream = (AlsaInfoSetStream_)audio_alsa_sym_(lib, "snd_pcm_info_set_stream", &ok);
        g_alsa_.pcm_info_get_name = (AlsaPcmGetName_)audio_alsa_sym_(lib, "snd_pcm_info_get_name", &ok);
        if (!ok) return 0;
        g_alsa_.set_error_handler(audio_alsa_quiet_);
    }
    g_alsa_.loaded = 1;
    return 1;
}

static const &char audio_alsa_strerror_(int err) {
    unsafe {
        const char* s = g_alsa_.strerror(err);
        if (s == (const char*)0) return "unknown ALSA error";
        return (const &char)s;
    }
}

static int audio_alsa_fail_(&AudioError e, int code, const &char what, int err) {
    return audio_fail2_(e, code, what, audio_alsa_strerror_(err));
}

static int audio_alsa_format_of_(unsigned int f) {
    if (f == AUDIO_FORMAT_I8) return ALSA_FMT_S8;
    if (f == AUDIO_FORMAT_I16) return ALSA_FMT_S16_LE;
    if (f == AUDIO_FORMAT_I24) return ALSA_FMT_S24_3LE;
    if (f == AUDIO_FORMAT_I32) return ALSA_FMT_S32_LE;
    if (f == AUDIO_FORMAT_F32) return ALSA_FMT_FLOAT_LE;
    if (f == AUDIO_FORMAT_F64) return ALSA_FMT_FLOAT64_LE;
    return -1;
}

static int audio_alsa_available_() { return audio_alsa_load_(); }

// One direction of a device: max channels, formats, rates. 0 if it can't
// be opened that way.
static int audio_alsa_probe_dir_(const &char name, int stream, &AudioDeviceInfo info) {
    SndPcm_* pcm = (SndPcm_*)0;
    int rc = 0;
    unsafe { rc = g_alsa_.pcm_open(&pcm, name, stream, ALSA_NONBLOCK); }
    if (rc < 0) return 0;
    unsigned int maxch = 0U;
    unsafe {
        SndHwParams_* hw = (SndHwParams_*)alloc_zeroed_raw_((unsigned long)g_alsa_.hw_sizeof());
        if (hw == (SndHwParams_*)0) { g_alsa_.pcm_close(pcm); return 0; }
        if (g_alsa_.hw_any(pcm, hw) >= 0) {
            g_alsa_.hw_get_channels_max(hw, &maxch);
            if (maxch > 64U) maxch = 64U;      // "unbounded" plugins report a huge maximum
            unsigned int fmts[6];
            fmts[0] = AUDIO_FORMAT_I8; fmts[1] = AUDIO_FORMAT_I16; fmts[2] = AUDIO_FORMAT_I24;
            fmts[3] = AUDIO_FORMAT_I32; fmts[4] = AUDIO_FORMAT_F32; fmts[5] = AUDIO_FORMAT_F64;
            for (int i = 0; i < 6; i = i + 1)
                if (g_alsa_.hw_test_format(pcm, hw, audio_alsa_format_of_(fmts[i])) == 0)
                    info.native_formats = info.native_formats | fmts[i];
            for (int i = 0; i < audio_standard_rate_count_(); i = i + 1) {
                unsigned int r = audio_standard_rate_(i);
                if (g_alsa_.hw_test_rate(pcm, hw, r, 0) == 0) audio_device_add_rate_(info, r);
            }
        }
        dealloc(hw);
        g_alsa_.pcm_close(pcm);
    }
    if (stream == ALSA_STREAM_PLAYBACK) info.output_channels = (int)maxch;
    else info.input_channels = (int)maxch;
    return maxch > 0U;
}

static void audio_alsa_finish_info_(&AudioDeviceInfo info) {
    info.duplex_channels = info.output_channels < info.input_channels ? info.output_channels : info.input_channels;
    if (info.supports_rate(48000U)) info.preferred_sample_rate = 48000U;
    else if (info.supports_rate(44100U)) info.preferred_sample_rate = 44100U;
    else if (info.sample_rate_count > 0) info.preferred_sample_rate = info.sample_rates[info.sample_rate_count - 1];
}

// Probes 'pcmName' and adds it when either direction opens.
static void audio_alsa_add_pcm_(&AudioHost_ h, const &char pcmName, const &char display, int isDefault) {
    struct AudioDeviceInfo info = audio_device_info_blank_();
    audio_strcpy_(&info.name[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, display);
    int o = audio_alsa_probe_dir_(pcmName, ALSA_STREAM_PLAYBACK, &info);
    int i = audio_alsa_probe_dir_(pcmName, ALSA_STREAM_CAPTURE, &info);
    if (!o && !i) return;
    audio_alsa_finish_info_(&info);
    info.is_default_output = isDefault && o;
    info.is_default_input = isDefault && i;
    match (audio_host_add_(h, &info, pcmName, 0LL)) {
        case some(r): {
            if (h.default_out == AUDIO_DEFAULT_DEVICE && o) h.default_out = r.info.id;
            if (h.default_in == AUDIO_DEFAULT_DEVICE && i) h.default_in = r.info.id;
        }
        case null: {}
    }
}

static int audio_alsa_probe_(&AudioHost_ h) {
    if (!audio_alsa_load_()) return audio_fail_(&h.error, AUDIO_ERR_UNAVAILABLE, "alsa: libasound.so.2 not found");
    audio_alsa_add_pcm_(h, "default", "Default ALSA Device", 1);
    int card = -1;
    unsafe {
        SndCtlCardInfo_* cinfo = (SndCtlCardInfo_*)alloc_zeroed_raw_((unsigned long)g_alsa_.ctl_card_info_sizeof());
        SndPcmInfo_* pinfo = (SndPcmInfo_*)alloc_zeroed_raw_((unsigned long)g_alsa_.pcm_info_sizeof());
        if (cinfo == (SndCtlCardInfo_*)0 || pinfo == (SndPcmInfo_*)0) {
            dealloc(cinfo); dealloc(pinfo);
            return audio_fail_(&h.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory");
        }
        while (g_alsa_.card_next(&card) == 0 && card >= 0) {
            char ctlName[32];
            audio_strcpy_(&ctlName[0], 32UL, "hw:");
            audio_strcat_int_(&ctlName[0], 32UL, (long long)card);
            SndCtl_* ctl = (SndCtl_*)0;
            if (g_alsa_.ctl_open(&ctl, (const char*)&ctlName[0], 0) < 0) continue;
            char cardName[80];
            audio_strcpy_(&cardName[0], 80UL, &ctlName[0]);
            if (g_alsa_.ctl_card_info(ctl, cinfo) == 0)
                audio_strcpy_(&cardName[0], 80UL, (const ?&char)g_alsa_.card_info_get_name(cinfo));
            int dev = -1;
            while (g_alsa_.ctl_pcm_next_device(ctl, &dev) == 0 && dev >= 0) {
                char pcmName[40];
                audio_strcpy_(&pcmName[0], 40UL, &ctlName[0]);
                audio_strcat_(&pcmName[0], 40UL, ",");
                audio_strcat_int_(&pcmName[0], 40UL, (long long)dev);
                char display[AUDIO_DEVICE_NAME_MAX];
                audio_strcpy_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, &cardName[0]);
                g_alsa_.pcm_info_set_device(pinfo, (unsigned int)dev);
                g_alsa_.pcm_info_set_subdevice(pinfo, 0U);
                g_alsa_.pcm_info_set_stream(pinfo, ALSA_STREAM_PLAYBACK);
                int got = g_alsa_.ctl_pcm_info(ctl, pinfo);
                if (got < 0) {
                    g_alsa_.pcm_info_set_stream(pinfo, ALSA_STREAM_CAPTURE);
                    got = g_alsa_.ctl_pcm_info(ctl, pinfo);
                }
                if (got == 0) {
                    audio_strcat_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, ": ");
                    audio_strcat_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX,
                                  (const ?&char)g_alsa_.pcm_info_get_name(pinfo));
                }
                audio_strcat_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, " (");
                audio_strcat_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, &pcmName[0]);
                audio_strcat_(&display[0], (unsigned long)AUDIO_DEVICE_NAME_MAX, ")");
                audio_alsa_add_pcm_(h, &pcmName[0], &display[0], 0);
            }
            g_alsa_.ctl_close(ctl);
        }
        dealloc(cinfo);
        dealloc(pinfo);
    }
    return AUDIO_OK;
}

static void audio_alsa_host_close_(&AudioHost_ h) { }

// ── streams ──────────────────────────────────────────────────────────────

struct AlsaStream_ {
    SndPcm_* pcm[2];
    unsigned char** chan_ptrs[2];   // non-interleaved access: one pointer per device channel
    int linked;
    size_t period;
    size_t buffer;
};

static ?&AlsaStream_ audio_alsa_of_(&AudioStreamState_ st) {
    unsafe { return (?&AlsaStream_)st.backend; }
}

// Configures one PCM. Sets the stream's (be) fields for 'dir'.
static int audio_alsa_setup_(&AudioStreamState_ st, &AlsaStream_ a, int dir, const &char name, int allowPlug) {
    int stream = dir == AUDIO_DIR_OUT ? ALSA_STREAM_PLAYBACK : ALSA_STREAM_CAPTURE;
    SndPcm_* pcm = (SndPcm_*)0;
    int rc = 0;
    unsafe { rc = g_alsa_.pcm_open(&pcm, name, stream, ALSA_NONBLOCK); }
    if (rc < 0) {
        int code = rc == 0 - ALSA_EBUSY ? AUDIO_ERR_DEVICE_BUSY : AUDIO_ERR_SYSTEM;
        return audio_alsa_fail_(&st.error, code, "alsa: snd_pcm_open", rc);
    }
    unsafe { g_alsa_.pcm_nonblock(pcm, 0); }
    SndHwParams_* hw = (SndHwParams_*)0;
    unsafe { hw = (SndHwParams_*)alloc_zeroed_raw_((unsigned long)g_alsa_.hw_sizeof()); }
    if (hw == (SndHwParams_*)0) {
        unsafe { g_alsa_.pcm_close(pcm); }
        return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory");
    }
    int result = AUDIO_OK;
    unsafe {
        while (1) {
            if ((rc = g_alsa_.hw_any(pcm, hw)) < 0) { result = audio_alsa_fail_(&st.error, AUDIO_ERR_SYSTEM, "alsa: hw_params_any", rc); break; }
            // Access: interleaved if the device can, else non-interleaved.
            int inter = 1;
            if (g_alsa_.hw_set_access(pcm, hw, ALSA_ACCESS_RW_INTERLEAVED) < 0) {
                if ((rc = g_alsa_.hw_set_access(pcm, hw, ALSA_ACCESS_RW_NONINTERLEAVED)) < 0) {
                    result = audio_alsa_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "alsa: no read/write access", rc);
                    break;
                }
                inter = 0;
            }
            st.device_interleaved[dir] = inter;
            // Format: the stream's, else the best the device takes.
            unsigned int order[7];
            order[0] = st.user_format; order[1] = AUDIO_FORMAT_F32; order[2] = AUDIO_FORMAT_I32;
            order[3] = AUDIO_FORMAT_I24; order[4] = AUDIO_FORMAT_I16; order[5] = AUDIO_FORMAT_F64;
            order[6] = AUDIO_FORMAT_I8;
            unsigned int chosen = 0U;
            for (int i = 0; i < 7; i = i + 1) {
                if (g_alsa_.hw_test_format(pcm, hw, audio_alsa_format_of_(order[i])) == 0) { chosen = order[i]; break; }
            }
            if (chosen == 0U) { result = audio_fail_(&st.error, AUDIO_ERR_UNSUPPORTED, "alsa: no supported sample format"); break; }
            g_alsa_.hw_set_format(pcm, hw, audio_alsa_format_of_(chosen));
            st.device_format[dir] = chosen;
            unsigned int ch = (unsigned int)(st.user_channels[dir] + st.first_channel[dir]);
            unsigned int minch = 0U;
            g_alsa_.hw_get_channels_min(hw, &minch);
            if (ch < minch) ch = minch;      // e.g. a hw device that's always 2-channel
            if ((rc = g_alsa_.hw_set_channels(pcm, hw, ch)) < 0) {
                result = audio_alsa_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "alsa: channel count not supported", rc);
                break;
            }
            st.device_channels[dir] = (int)ch;
            // Rate: exact. A hw: device that can't is retried through plug.
            if (g_alsa_.hw_test_rate(pcm, hw, st.sample_rate, 0) < 0) {
                result = AUDIO_ERR_INVALID_PARAM;
                audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "alsa: sample rate not supported by the device");
                break;
            }
            unsigned int rate = st.sample_rate;
            int sub = 0;
            g_alsa_.hw_set_rate_near(pcm, hw, &rate, &sub);
            // Period (the callback size) and the number of periods.
            size_t period = (size_t)(st.buffer_frames == 0U ? 512U : st.buffer_frames);
            if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U) period = (size_t)64;
            if (dir == AUDIO_DIR_IN && st.mode_out) period = a.period;   // duplex: the playback period
            sub = 0;
            g_alsa_.hw_set_period_size_near(pcm, hw, &period, &sub);
            unsigned int periods = st.buffer_count == 0U ? 2U : st.buffer_count;
            if ((st.flags & AUDIO_FLAG_MINIMIZE_LATENCY) != 0U) periods = 2U;
            sub = 0;
            g_alsa_.hw_set_periods_near(pcm, hw, &periods, &sub);
            if ((rc = g_alsa_.hw_params(pcm, hw)) < 0) { result = audio_alsa_fail_(&st.error, AUDIO_ERR_SYSTEM, "alsa: hw_params", rc); break; }
            size_t got = (size_t)0;
            size_t buf = (size_t)0;
            sub = 0;
            g_alsa_.hw_get_period_size(hw, &got, &sub);
            g_alsa_.hw_get_buffer_size(hw, &buf);
            if (dir == AUDIO_DIR_IN && st.mode_out && got != a.period) {
                result = audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "alsa: duplex devices disagree on the period size");
                break;
            }
            a.period = got;
            a.buffer = buf;
            st.buffer_frames = (unsigned int)got;
            st.buffer_count = periods;
            st.latency[dir] = (long)buf;
            break;
        }
        dealloc(hw);
    }
    if (result == AUDIO_OK) {
        // Software params: playback starts once a full buffer is queued,
        // wake-ups once a period is free.
        unsafe {
            SndSwParams_* sw = (SndSwParams_*)alloc_zeroed_raw_((unsigned long)g_alsa_.sw_sizeof());
            if (sw != (SndSwParams_*)0) {
                g_alsa_.sw_current(pcm, sw);
                g_alsa_.sw_set_start_threshold(pcm, sw, dir == AUDIO_DIR_OUT ? a.buffer : (size_t)1);
                g_alsa_.sw_set_avail_min(pcm, sw, a.period);
                g_alsa_.sw_params(pcm, sw);
                dealloc(sw);
            }
        }
        a.pcm[dir] = pcm;
        return AUDIO_OK;
    }
    unsafe { g_alsa_.pcm_close(pcm); }
    if (result == AUDIO_ERR_INVALID_PARAM && allowPlug) {
        // hw:X,Y can't do the rate: ALSA's plug layer converts.
        if (audio_strprefix_(name, "hw:")) {
            char plug[AUDIO_DEVICE_KEY_MAX];
            audio_strcpy_(&plug[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, "plug");
            audio_strcat_(&plug[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, name);
            return audio_alsa_setup_(st, a, dir, &plug[0], 0);
        }
    }
    return result;
}

static void audio_alsa_release_(&AlsaStream_ a) {
    unsafe {
        for (int d = 0; d < 2; d = d + 1) {
            if (a.pcm[d] != (SndPcm_*)0) g_alsa_.pcm_close(a.pcm[d]);
            a.pcm[d] = (SndPcm_*)0;
            dealloc(a.chan_ptrs[d]);
            a.chan_ptrs[d] = (unsigned char**)0;
        }
    }
}

static int audio_alsa_open_(&AudioHost_ h, &AudioStreamState_ st) {
    if (!audio_alsa_load_()) return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "alsa: libasound.so.2 not found");
    ?&AlsaStream_ ap = null;
    unsafe { ap = (?&AlsaStream_)alloc_zeroed_raw_(sizeof(struct AlsaStream_)); }
    match (ap) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory"); }
        case some(a): {
            int allowPlug = (st.flags & AUDIO_FLAG_EXCLUSIVE) == 0U;
            int rc = AUDIO_OK;
            for (int d = 0; d < 2 && rc == AUDIO_OK; d = d + 1) {
                int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
                if (!on) continue;
                char name[AUDIO_DEVICE_KEY_MAX];
                if (st.device_name[0] != (char)0) audio_strcpy_(&name[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, &st.device_name[0]);
                else if ((st.flags & AUDIO_FLAG_ALSA_USE_DEFAULT) != 0U) audio_strcpy_(&name[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, "default");
                else audio_strcpy_(&name[0], (unsigned long)AUDIO_DEVICE_KEY_MAX, &st.device_key[d][0]);
                rc = audio_alsa_setup_(st, a, d, &name[0], allowPlug);
            }
            if (rc != AUDIO_OK) {
                audio_alsa_release_(a);
                unsafe { dealloc((?&heap AlsaStream_)a); }
                return rc;
            }
            st.max_frames = st.buffer_frames;
            // Duplex: start/stop the two PCMs together when ALSA can.
            unsafe {
                if (st.mode_out && st.mode_in && g_alsa_.link(a.pcm[0], a.pcm[1]) == 0) a.linked = 1;
                st.backend = (?&heap AudioStreamImpl_)a;
            }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

// Builds the per-channel pointer arrays for non-interleaved access (after
// audio.sc allocated device_buf).
static int audio_alsa_chan_ptrs_(&AudioStreamState_ st, &AlsaStream_ a) {
    for (int d = 0; d < 2; d = d + 1) {
        int on = d == AUDIO_DIR_OUT ? st.mode_out : st.mode_in;
        if (!on || st.device_interleaved[d] || a.chan_ptrs[d] != (unsigned char**)0) continue;
        unsafe {
            unsigned long n = (unsigned long)st.device_channels[d];
            a.chan_ptrs[d] = (unsigned char**)alloc_zeroed_raw_(checked_mul_size(n, sizeof(unsigned char*)));
            if (a.chan_ptrs[d] == (unsigned char**)0) return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory");
            unsigned long stride = (unsigned long)st.max_frames * (unsigned long)audio_format_bytes(st.device_format[d]);
            unsigned char* base = (unsigned char*)st.device_buf[d].ptr;
            for (unsigned long c = 0UL; c < n; c = c + 1UL) a.chan_ptrs[d][c] = base + c * stride;
        }
    }
    return AUDIO_OK;
}

// Writes one period (all of it, recovering from underruns). <0: fatal.
static int audio_alsa_write_(&AudioStreamState_ st, &AlsaStream_ a, unsigned long frames) {
    unsigned long done = 0UL;
    unsigned long fb = audio_stream_device_frame_bytes_(st, AUDIO_DIR_OUT);
    unsigned long sb = (unsigned long)audio_format_bytes(st.device_format[AUDIO_DIR_OUT]);
    int tries = 0;
    while (done < frames) {
        ssize_t r = (ssize_t)0;
        unsafe {
            if (st.device_interleaved[AUDIO_DIR_OUT]) {
                const unsigned char* p = (const unsigned char*)st.device_buf[AUDIO_DIR_OUT].ptr;
                r = g_alsa_.writei(a.pcm[AUDIO_DIR_OUT], p + done * fb, (size_t)(frames - done));
            } else {
                // Offsets into each channel plane.
                unsigned char* ptrs[64];
                int n = st.device_channels[AUDIO_DIR_OUT];
                if (n > 64) n = 64;
                for (int c = 0; c < n; c = c + 1)
                    ptrs[c] = a.chan_ptrs[AUDIO_DIR_OUT][c] + done * sb;
                r = g_alsa_.writen(a.pcm[AUDIO_DIR_OUT], &ptrs[0], (size_t)(frames - done));
            }
        }
        if ((long long)r >= 0LL) { done = done + (unsigned long)r; continue; }
        int err = (int)r;
        if (err == 0 - ALSA_EAGAIN) continue;
        if (err == 0 - ALSA_EPIPE) audio_stream_flag_status_(st, AUDIO_STATUS_OUTPUT_UNDERFLOW);
        tries = tries + 1;
        int rc = 0;
        unsafe { rc = g_alsa_.recover(a.pcm[AUDIO_DIR_OUT], err, 1); }
        if (rc < 0 || tries > 8) {
            audio_stream_set_rt_error_(st, AUDIO_ERR_SYSTEM, audio_alsa_strerror_(err));
            return -1;
        }
    }
    return 0;
}

// Reads one period. <0: fatal.
static int audio_alsa_read_(&AudioStreamState_ st, &AlsaStream_ a, unsigned long frames) {
    unsigned long done = 0UL;
    unsigned long fb = audio_stream_device_frame_bytes_(st, AUDIO_DIR_IN);
    unsigned long sb = (unsigned long)audio_format_bytes(st.device_format[AUDIO_DIR_IN]);
    int tries = 0;
    while (done < frames) {
        ssize_t r = (ssize_t)0;
        unsafe {
            if (st.device_interleaved[AUDIO_DIR_IN]) {
                unsigned char* p = (unsigned char*)st.device_buf[AUDIO_DIR_IN].ptr;
                r = g_alsa_.readi(a.pcm[AUDIO_DIR_IN], p + done * fb, (size_t)(frames - done));
            } else {
                unsigned char* ptrs[64];
                int n = st.device_channels[AUDIO_DIR_IN];
                if (n > 64) n = 64;
                for (int c = 0; c < n; c = c + 1)
                    ptrs[c] = a.chan_ptrs[AUDIO_DIR_IN][c] + done * sb;
                r = g_alsa_.readn(a.pcm[AUDIO_DIR_IN], &ptrs[0], (size_t)(frames - done));
            }
        }
        if ((long long)r >= 0LL) { done = done + (unsigned long)r; continue; }
        int err = (int)r;
        if (err == 0 - ALSA_EAGAIN) continue;
        if (err == 0 - ALSA_EPIPE) audio_stream_flag_status_(st, AUDIO_STATUS_INPUT_OVERFLOW);
        tries = tries + 1;
        int rc = 0;
        unsafe {
            rc = g_alsa_.recover(a.pcm[AUDIO_DIR_IN], err, 1);
            if (rc >= 0 && !a.linked) g_alsa_.pcm_start(a.pcm[AUDIO_DIR_IN]);
        }
        if (rc < 0 || tries > 8) {
            audio_stream_set_rt_error_(st, AUDIO_ERR_SYSTEM, audio_alsa_strerror_(err));
            return -1;
        }
    }
    return 0;
}

static void audio_alsa_thread_(&AudioStreamState_ st) {
    match (audio_alsa_of_(st)) {
        case some(a): {
            unsigned long period = (unsigned long)a.period;
            int how = AUDIO_CB_CONTINUE;
            int failed = 0;
            while (1) {
                how = audio_stream_stop_requested_(st);
                if (how != AUDIO_CB_CONTINUE) break;
                if (st.mode_in && audio_alsa_read_(st, a, period) < 0) { failed = 1; break; }
                int r = audio_stream_process_(st, (unsigned int)period);
                if (r == AUDIO_CB_ABORT) { how = r; break; }
                if (st.mode_out && audio_alsa_write_(st, a, period) < 0) { failed = 1; break; }
                if (r == AUDIO_CB_STOP) { how = r; break; }
            }
            unsafe {
                if (st.mode_out) {
                    if (how == AUDIO_CB_STOP && !failed) g_alsa_.pcm_drain(a.pcm[AUDIO_DIR_OUT]);
                    else g_alsa_.pcm_drop(a.pcm[AUDIO_DIR_OUT]);
                }
                if (st.mode_in && !a.linked) g_alsa_.pcm_drop(a.pcm[AUDIO_DIR_IN]);
            }
        }
        case null: {}
    }
    audio_stream_mark_stopped_(st, null);
}

static int audio_alsa_start_(&AudioStreamState_ st) {
    match (audio_alsa_of_(st)) {
        case some(a): {
            int rc = audio_alsa_chan_ptrs_(st, a);
            if (rc != AUDIO_OK) return rc;
            unsafe {
                for (int d = 0; d < 2; d = d + 1) {
                    if (a.pcm[d] == (SndPcm_*)0) continue;
                    if (d == AUDIO_DIR_IN && a.linked) continue;   // prepared with its link partner
                    int e = g_alsa_.pcm_prepare(a.pcm[d]);
                    if (e < 0) return audio_alsa_fail_(&st.error, AUDIO_ERR_SYSTEM, "alsa: snd_pcm_prepare", e);
                }
            }
            if (st.mode_out) {
                // Pre-roll silence: all but one period for playback alone (the
                // device starts when the first callback's period completes the
                // buffer); the whole buffer in duplex, so playback — and with
                // it a linked capture — is running before the first read.
                audio_stream_silence_(st, st.buffer_frames);
                unsigned long pre = (unsigned long)a.buffer;
                if (!st.mode_in) pre = pre - (unsigned long)a.period;
                while (pre > 0UL) {
                    unsigned long n = pre > (unsigned long)a.period ? (unsigned long)a.period : pre;
                    if (audio_alsa_write_(st, a, n) < 0) return audio_fail_(&st.error, AUDIO_ERR_SYSTEM, "alsa: pre-roll failed");
                    pre = pre - n;
                }
            }
            if (st.mode_in && !(a.linked && st.mode_out)) {
                unsafe { g_alsa_.pcm_start(a.pcm[AUDIO_DIR_IN]); }
            }
            return audio_stream_thread_start_(st, audio_alsa_thread_);
        }
        case null: {}
    }
    return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "alsa: stream not open");
}

static int audio_alsa_stop_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    return AUDIO_OK;
}

static void audio_alsa_close_(&AudioStreamState_ st) {
    audio_stream_thread_join_(st);
    match (audio_alsa_of_(st)) {
        case some(a): {
            audio_alsa_release_(a);
            unsafe { dealloc((?&heap AlsaStream_)a); }
        }
        case null: {}
    }
    unsafe { st.backend = null; }
}

static struct AudioBackendVt_ g_audio_alsa_vt_;
static int g_audio_alsa_vt_ready_ = 0;

&static AudioBackendVt_ audio_alsa_backend_() {
    if (!g_audio_alsa_vt_ready_) {
        g_audio_alsa_vt_.api = AUDIO_API_ALSA;
        g_audio_alsa_vt_.available = audio_alsa_available_;
        g_audio_alsa_vt_.probe = audio_alsa_probe_;
        g_audio_alsa_vt_.host_close = audio_alsa_host_close_;
        g_audio_alsa_vt_.open = audio_alsa_open_;
        g_audio_alsa_vt_.start = audio_alsa_start_;
        g_audio_alsa_vt_.stop = audio_alsa_stop_;
        g_audio_alsa_vt_.abort = audio_alsa_stop_;
        g_audio_alsa_vt_.close = audio_alsa_close_;
        g_audio_alsa_vt_ready_ = 1;
    }
    return &g_audio_alsa_vt_;
}

} // namespace std
