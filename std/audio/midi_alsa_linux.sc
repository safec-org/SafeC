// SafeC Standard Library — MIDI backend: the ALSA sequencer (Linux). See midi.h.
//
// libasound.so.2 is loaded with dlopen() on first use; without it, or
// without a sequencer (/dev/snd/seq: the snd-seq kernel module), the API
// reports AUDIO_ERR_UNAVAILABLE. Ports are other sequencer clients' MIDI
// ports (hardware, software synths, other programs), named
// "<client>:<port> <client id>:<port id>" as RtMidi and aconnect show them.
// open_port() creates our own port and subscribes it to/from the chosen
// one; open_virtual_port() creates a port other clients subscribe to.
//
// Input runs on a thread std starts per port: poll() on the sequencer's
// descriptors, snd_seq_event_input(), snd_midi_event_decode() back to MIDI
// bytes (no running status), then std's parser (which reassembles sysex
// the sequencer delivers in chunks). Output encodes each message with
// snd_midi_event_encode() and sends it directly (no queue).
#pragma once
#include <std/stddef.h>
#include <std/audio/midi_backend.h>
#include <std/audio/audio_sys.h>
#include <std/mem.h>
#include <std/thread.h>

#define ASEQ_OPEN_OUTPUT 1
#define ASEQ_OPEN_INPUT  2
#define ASEQ_OPEN_DUPLEX 3
#define ASEQ_NONBLOCK    1
#define ASEQ_CAP_READ       1U
#define ASEQ_CAP_WRITE      2U
#define ASEQ_CAP_SUBS_READ  32U
#define ASEQ_CAP_SUBS_WRITE 64U
#define ASEQ_CAP_NO_EXPORT  128U
#define ASEQ_TYPE_MIDI_GENERIC 2U
#define ASEQ_TYPE_SYNTH        1024U
#define ASEQ_TYPE_APPLICATION  1048576U
#define ASEQ_ADDRESS_UNKNOWN     253
#define ASEQ_ADDRESS_SUBSCRIBERS 254
#define ASEQ_QUEUE_DIRECT        253
#define ASEQ_POLLIN  1
#define ASEQ_EAGAIN  11
#define ASEQ_ENOSPC  28

namespace std {

// snd_seq_event_t: 28 bytes (type, flags, tag, queue; time; source; dest; data).
struct AlsaSeqEvent_ {
    unsigned char type;
    unsigned char flags;
    unsigned char tag;
    unsigned char queue;
    unsigned int  time_a;
    unsigned int  time_b;
    unsigned char source_client;
    unsigned char source_port;
    unsigned char dest_client;
    unsigned char dest_port;
    unsigned char data[12];
};

static_assert(sizeof(struct AlsaSeqEvent_) == 28, "snd_seq_event_t is 28 bytes");

typedef fn int(void**, const char*, int, int) AseqOpen_;
typedef fn int(void*) AseqClose_;
typedef fn int(void*, const char*) AseqSetName_;
typedef fn int(void*) AseqClientId_;
typedef fn int(void*, const char*, unsigned int, unsigned int) AseqCreatePort_;
typedef fn int(void*, int) AseqDeletePort_;
typedef fn int(void*, int, int, int) AseqConnect_;
typedef fn size_t() AseqSizeof_;
typedef fn void(void*, int) AseqInfoSetInt_;
typedef fn int(const void*) AseqInfoGetInt_;
typedef fn unsigned int(const void*) AseqInfoGetUint_;
typedef fn const char*(const void*) AseqInfoGetName_;
typedef fn int(void*, void*) AseqQueryNext_;
typedef fn int(void*, short) AseqPollCount_;
typedef fn int(void*, void*, unsigned int, short) AseqPollDesc_;
typedef fn int(void*, &?&AlsaSeqEvent_) AseqEventInput_;
typedef fn int(void*, &AlsaSeqEvent_) AseqEventOutput_;
typedef fn int(void*, int) AseqNonblock_;
typedef fn int(void*) AseqSetHandler_;
typedef fn int(size_t, void**) AmevNew_;
typedef fn void(void*) AmevFree_;
typedef fn int(void*, size_t) AmevResize_;
typedef fn void(void*) AmevReset_;
typedef fn long(void*, const unsigned char*, long, &AlsaSeqEvent_) AmevEncode_;
typedef fn long(void*, unsigned char*, long, const &AlsaSeqEvent_) AmevDecode_;
typedef fn void(void*, int) AmevNoStatus_;

extern int poll(void* fds, unsigned long nfds, int timeout);

struct AlsaSeqLib_ {
    int loaded;
    AseqOpen_ open;
    AseqClose_ close;
    AseqSetName_ set_client_name;
    AseqClientId_ client_id;
    AseqCreatePort_ create_simple_port;
    AseqDeletePort_ delete_simple_port;
    AseqConnect_ connect_from;
    AseqConnect_ connect_to;
    AseqSizeof_ client_info_sizeof;
    AseqSizeof_ port_info_sizeof;
    AseqInfoSetInt_ client_info_set_client;
    AseqInfoGetInt_ client_info_get_client;
    AseqInfoGetName_ client_info_get_name;
    AseqQueryNext_ query_next_client;
    AseqInfoSetInt_ port_info_set_client;
    AseqInfoSetInt_ port_info_set_port;
    AseqQueryNext_ query_next_port;
    AseqInfoGetUint_ port_info_get_capability;
    AseqInfoGetUint_ port_info_get_type;
    AseqInfoGetName_ port_info_get_name;
    AseqInfoGetInt_ port_info_get_port;
    AseqInfoGetInt_ port_info_get_client;
    AseqPollCount_ poll_descriptors_count;
    AseqPollDesc_ poll_descriptors;
    AseqEventInput_ event_input;
    AseqEventOutput_ event_output_direct;
    AseqNonblock_ nonblock;
    AseqSetHandler_ set_error_handler;
    AmevNew_ mev_new;
    AmevFree_ mev_free;
    AmevResize_ mev_resize;
    AmevReset_ mev_reset_encode;
    AmevReset_ mev_reset_decode;
    AmevEncode_ mev_encode;
    AmevDecode_ mev_decode;
    AmevNoStatus_ mev_no_status;
};

static struct AlsaSeqLib_ g_aseq_;

static void midi_alsa_quiet_(const char* file, int line, const char* function, int err, const char* fmt) { }

static void* midi_alsa_sym_(?&void lib, const &char name, &int ok) {
    ?&void p = audio_dlsym_(lib, name);
    if (p.is_null()) *ok = 0;
    unsafe { return (void*)p; }
}

static int midi_alsa_load_() {
    if (g_aseq_.loaded != 0) return g_aseq_.loaded > 0;
    g_aseq_.loaded = -1;
    ?&void lib = audio_dlopen2_("libasound.so.2", "libasound.so");
    if (lib.is_null()) return 0;
    int ok = 1;
    unsafe {
        g_aseq_.open = (AseqOpen_)midi_alsa_sym_(lib, "snd_seq_open", &ok);
        g_aseq_.close = (AseqClose_)midi_alsa_sym_(lib, "snd_seq_close", &ok);
        g_aseq_.set_client_name = (AseqSetName_)midi_alsa_sym_(lib, "snd_seq_set_client_name", &ok);
        g_aseq_.client_id = (AseqClientId_)midi_alsa_sym_(lib, "snd_seq_client_id", &ok);
        g_aseq_.create_simple_port = (AseqCreatePort_)midi_alsa_sym_(lib, "snd_seq_create_simple_port", &ok);
        g_aseq_.delete_simple_port = (AseqDeletePort_)midi_alsa_sym_(lib, "snd_seq_delete_simple_port", &ok);
        g_aseq_.connect_from = (AseqConnect_)midi_alsa_sym_(lib, "snd_seq_connect_from", &ok);
        g_aseq_.connect_to = (AseqConnect_)midi_alsa_sym_(lib, "snd_seq_connect_to", &ok);
        g_aseq_.client_info_sizeof = (AseqSizeof_)midi_alsa_sym_(lib, "snd_seq_client_info_sizeof", &ok);
        g_aseq_.port_info_sizeof = (AseqSizeof_)midi_alsa_sym_(lib, "snd_seq_port_info_sizeof", &ok);
        g_aseq_.client_info_set_client = (AseqInfoSetInt_)midi_alsa_sym_(lib, "snd_seq_client_info_set_client", &ok);
        g_aseq_.client_info_get_client = (AseqInfoGetInt_)midi_alsa_sym_(lib, "snd_seq_client_info_get_client", &ok);
        g_aseq_.client_info_get_name = (AseqInfoGetName_)midi_alsa_sym_(lib, "snd_seq_client_info_get_name", &ok);
        g_aseq_.query_next_client = (AseqQueryNext_)midi_alsa_sym_(lib, "snd_seq_query_next_client", &ok);
        g_aseq_.port_info_set_client = (AseqInfoSetInt_)midi_alsa_sym_(lib, "snd_seq_port_info_set_client", &ok);
        g_aseq_.port_info_set_port = (AseqInfoSetInt_)midi_alsa_sym_(lib, "snd_seq_port_info_set_port", &ok);
        g_aseq_.query_next_port = (AseqQueryNext_)midi_alsa_sym_(lib, "snd_seq_query_next_port", &ok);
        g_aseq_.port_info_get_capability = (AseqInfoGetUint_)midi_alsa_sym_(lib, "snd_seq_port_info_get_capability", &ok);
        g_aseq_.port_info_get_type = (AseqInfoGetUint_)midi_alsa_sym_(lib, "snd_seq_port_info_get_type", &ok);
        g_aseq_.port_info_get_name = (AseqInfoGetName_)midi_alsa_sym_(lib, "snd_seq_port_info_get_name", &ok);
        g_aseq_.port_info_get_port = (AseqInfoGetInt_)midi_alsa_sym_(lib, "snd_seq_port_info_get_port", &ok);
        g_aseq_.port_info_get_client = (AseqInfoGetInt_)midi_alsa_sym_(lib, "snd_seq_port_info_get_client", &ok);
        g_aseq_.poll_descriptors_count = (AseqPollCount_)midi_alsa_sym_(lib, "snd_seq_poll_descriptors_count", &ok);
        g_aseq_.poll_descriptors = (AseqPollDesc_)midi_alsa_sym_(lib, "snd_seq_poll_descriptors", &ok);
        g_aseq_.event_input = (AseqEventInput_)midi_alsa_sym_(lib, "snd_seq_event_input", &ok);
        g_aseq_.event_output_direct = (AseqEventOutput_)midi_alsa_sym_(lib, "snd_seq_event_output_direct", &ok);
        g_aseq_.nonblock = (AseqNonblock_)midi_alsa_sym_(lib, "snd_seq_nonblock", &ok);
        g_aseq_.set_error_handler = (AseqSetHandler_)midi_alsa_sym_(lib, "snd_lib_error_set_handler", &ok);
        g_aseq_.mev_new = (AmevNew_)midi_alsa_sym_(lib, "snd_midi_event_new", &ok);
        g_aseq_.mev_free = (AmevFree_)midi_alsa_sym_(lib, "snd_midi_event_free", &ok);
        g_aseq_.mev_resize = (AmevResize_)midi_alsa_sym_(lib, "snd_midi_event_resize_buffer", &ok);
        g_aseq_.mev_reset_encode = (AmevReset_)midi_alsa_sym_(lib, "snd_midi_event_reset_encode", &ok);
        g_aseq_.mev_reset_decode = (AmevReset_)midi_alsa_sym_(lib, "snd_midi_event_reset_decode", &ok);
        g_aseq_.mev_encode = (AmevEncode_)midi_alsa_sym_(lib, "snd_midi_event_encode", &ok);
        g_aseq_.mev_decode = (AmevDecode_)midi_alsa_sym_(lib, "snd_midi_event_decode", &ok);
        g_aseq_.mev_no_status = (AmevNoStatus_)midi_alsa_sym_(lib, "snd_midi_event_no_status", &ok);
    }
    if (!ok) return 0;
    // ALSA prints errors to stderr (no sequencer, ...); std reports them.
    unsafe { g_aseq_.set_error_handler((void*)midi_alsa_quiet_); }
    g_aseq_.loaded = 1;
    return 1;
}

struct AlsaSeqClient_ {
    void* seq;
    int client_id;
    int port;                 // our port, -1 if none
    void* coder;              // snd_midi_event_t (encode for output, decode for input)
    unsigned long coder_size;
    int stop;                 // (atomic) input thread: exit
    unsigned long long thread;
    int thread_live;
    unsigned char* decode_buf;
    struct MidiPortState_* st;
};

static int midi_alsa_available_() {
    if (!midi_alsa_load_()) return 0;
    void* seq = (void*)0;
    int rc = 0;
    unsafe { rc = g_aseq_.open(&seq, "default", ASEQ_OPEN_OUTPUT, ASEQ_NONBLOCK); }
    if (rc < 0) return 0;
    unsafe { g_aseq_.close(seq); }
    return 1;
}

static ?&AlsaSeqClient_ midi_alsa_of_(&MidiPortState_ st) {
    unsafe { return (?&AlsaSeqClient_)st.backend; }
}

static int midi_alsa_client_open_(&MidiPortState_ st) {
    if (!midi_alsa_load_()) return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "alsa: libasound.so.2 not found");
    ?&AlsaSeqClient_ cp = null;
    unsafe { cp = (?&AlsaSeqClient_)alloc_zeroed_raw_(sizeof(struct AlsaSeqClient_)); }
    match (cp) {
        case null: { return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory"); }
        case some(c): {
            c.port = -1;
            int rc = 0;
            unsafe {
                c.st = (struct MidiPortState_*)st;
                rc = g_aseq_.open(&c.seq, "default", st.is_input ? ASEQ_OPEN_DUPLEX : ASEQ_OPEN_OUTPUT,
                                  st.is_input ? ASEQ_NONBLOCK : 0);
            }
            if (rc < 0) {
                unsafe { dealloc_raw_((void*)c); }
                return audio_fail_(&st.error, AUDIO_ERR_UNAVAILABLE, "alsa: no MIDI sequencer (/dev/snd/seq)");
            }
            unsafe {
                g_aseq_.set_client_name(c.seq, (const char*)&st.client_name[0]);
                c.client_id = g_aseq_.client_id(c.seq);
                c.coder_size = st.is_input ? 256UL : 32UL;
                if (g_aseq_.mev_new((size_t)c.coder_size, &c.coder) < 0) c.coder = (void*)0;
                if (st.is_input) {
                    c.decode_buf = (unsigned char*)alloc_zeroed_raw_(MIDI_MAX_SYSEX);
                    if (c.coder != (void*)0) g_aseq_.mev_no_status(c.coder, 1);
                }
                if (c.coder == (void*)0 || (st.is_input && c.decode_buf == (unsigned char*)0)) {
                    if (c.coder != (void*)0) g_aseq_.mev_free(c.coder);
                    dealloc_raw_((void*)c.decode_buf);
                    g_aseq_.close(c.seq);
                    dealloc_raw_((void*)c);
                    return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: out of memory");
                }
                st.backend = (void*)c;
            }
            return AUDIO_OK;
        }
    }
    return AUDIO_ERR_INVALID_USE;
}

// The 'index'-th port usable in our direction (index < 0: count them).
// Fills client/port/name when found; returns the count when counting, 1/0 otherwise.
static int midi_alsa_port_at_(&AlsaSeqClient_ c, int input, int index, &int outClient, &int outPort,
                              &char name, unsigned long cap) {
    int count = 0;
    unsafe {
        void* cinfo = alloc_zeroed_raw_((unsigned long)g_aseq_.client_info_sizeof());
        void* pinfo = alloc_zeroed_raw_((unsigned long)g_aseq_.port_info_sizeof());
        if (cinfo == (void*)0 || pinfo == (void*)0) {
            dealloc_raw_(cinfo); dealloc_raw_(pinfo);
            return 0;
        }
        unsigned int want = input ? (ASEQ_CAP_READ | ASEQ_CAP_SUBS_READ) : (ASEQ_CAP_WRITE | ASEQ_CAP_SUBS_WRITE);
        g_aseq_.client_info_set_client(cinfo, -1);
        int found = 0;
        while (!found && g_aseq_.query_next_client(c.seq, cinfo) >= 0) {
            int client = g_aseq_.client_info_get_client((const void*)cinfo);
            if (client == 0 || client == c.client_id) continue;   // the System client, ourselves
            g_aseq_.port_info_set_client(pinfo, client);
            g_aseq_.port_info_set_port(pinfo, -1);
            while (g_aseq_.query_next_port(c.seq, pinfo) >= 0) {
                unsigned int caps = g_aseq_.port_info_get_capability((const void*)pinfo);
                unsigned int type = g_aseq_.port_info_get_type((const void*)pinfo);
                if ((type & (ASEQ_TYPE_MIDI_GENERIC | ASEQ_TYPE_SYNTH | ASEQ_TYPE_APPLICATION)) == 0U) continue;
                if ((caps & want) != want || (caps & ASEQ_CAP_NO_EXPORT) != 0U) continue;
                if (count == index) {
                    *outClient = client;
                    *outPort = g_aseq_.port_info_get_port((const void*)pinfo);
                    audio_strcpy_(name, cap, (const ?&char)g_aseq_.client_info_get_name((const void*)cinfo));
                    audio_strcat_(name, cap, ":");
                    audio_strcat_(name, cap, (const ?&char)g_aseq_.port_info_get_name((const void*)pinfo));
                    audio_strcat_(name, cap, " ");
                    audio_strcat_int_(name, cap, (long long)client);
                    audio_strcat_(name, cap, ":");
                    audio_strcat_int_(name, cap, (long long)*outPort);
                    found = 1;
                    break;
                }
                count = count + 1;
            }
        }
        dealloc_raw_(cinfo);
        dealloc_raw_(pinfo);
        if (index < 0) return count;
        return found;
    }
}

static unsigned int midi_alsa_port_count_(&MidiPortState_ st) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            int cl = 0; int pt = 0;
            char name[8];
            int n = midi_alsa_port_at_(c, st.is_input, -1, &cl, &pt, &name[0], 8UL);
            return n < 0 ? 0U : (unsigned int)n;
        }
        case null: {}
    }
    return 0U;
}

static int midi_alsa_port_name_(&MidiPortState_ st, unsigned int index, &char out, unsigned long cap) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            int cl = 0; int pt = 0;
            if (midi_alsa_port_at_(c, st.is_input, (int)index, &cl, &pt, out, cap)) return AUDIO_OK;
            return AUDIO_ERR_INVALID_DEVICE;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

// The input thread.
static ?&AlsaSeqClient_ midi_alsa_thread_(?&AlsaSeqClient_ arg) {
    match (arg) {
        case some(c): {
            unsafe {
                struct MidiPortState_* st = c.st;
                int n = g_aseq_.poll_descriptors_count(c.seq, (short)ASEQ_POLLIN);
                if (n <= 0) return null;
                if (n > 8) n = 8;
                unsigned long long pfds[8];        // struct pollfd { int fd; short events, revents; }: 8 bytes
                g_aseq_.poll_descriptors(c.seq, (void*)&pfds[0], (unsigned int)n, (short)ASEQ_POLLIN);
                while (atomic_load((_Atomic int*)&c.stop) == 0) {
                    poll((void*)&pfds[0], (unsigned long)n, 50);
                    while (1) {
                        ?&AlsaSeqEvent_ ev = null;
                        int r = g_aseq_.event_input(c.seq, &ev);
                        if (r == 0 - ASEQ_EAGAIN) break;
                        if (r == 0 - ASEQ_ENOSPC) {   // the kernel's input queue overran
                            atomic_fetch_add((_Atomic unsigned long long*)&st->dropped_count, 1ULL);
                            continue;
                        }
                        if (r < 0) break;
                        match (ev) {
                            case some(e): {
                                long got = g_aseq_.mev_decode(c.coder, c.decode_buf, (long)MIDI_MAX_SYSEX, e);
                                if (got > 0L)
                                    midi_in_feed_((&MidiPortState_)st, (const &unsigned char)c.decode_buf, (unsigned long)got, -1.0);
                                g_aseq_.mev_reset_decode(c.coder);
                            }
                            case null: {}
                        }
                    }
                }
            }
        }
        case null: {}
    }
    return null;
}

static int midi_alsa_make_port_(&MidiPortState_ st, &AlsaSeqClient_ c, const &char name) {
    unsigned int caps = st.is_input ? (ASEQ_CAP_WRITE | ASEQ_CAP_SUBS_WRITE) : (ASEQ_CAP_READ | ASEQ_CAP_SUBS_READ);
    int p = 0;
    unsafe { p = g_aseq_.create_simple_port(c.seq, name, caps, ASEQ_TYPE_MIDI_GENERIC | ASEQ_TYPE_APPLICATION); }
    if (p < 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "alsa: could not create a sequencer port", (long long)p);
    c.port = p;
    return AUDIO_OK;
}

static int midi_alsa_start_input_(&MidiPortState_ st, &AlsaSeqClient_ c) {
    if (!st.is_input) return AUDIO_OK;
    unsafe { atomic_store((_Atomic int*)&c.stop, 0); }
    unsigned long long tid = 0ULL;
    int rc = 0;
    unsafe { rc = thread_create(&tid, midi_alsa_thread_, (?&AlsaSeqClient_)c); }
    if (rc != 0) return audio_fail_(&st.error, AUDIO_ERR_THREAD, "alsa: could not start the MIDI input thread");
    c.thread = tid;
    c.thread_live = 1;
    return AUDIO_OK;
}

static void midi_alsa_drop_port_(&AlsaSeqClient_ c) {
    if (c.thread_live) {
        unsafe { atomic_store((_Atomic int*)&c.stop, 1); }
        thread_join(c.thread);
        c.thread_live = 0;
    }
    if (c.port >= 0) {
        unsafe { g_aseq_.delete_simple_port(c.seq, c.port); }
        c.port = -1;
    }
}

static int midi_alsa_open_port_(&MidiPortState_ st, unsigned int index, const &char name) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            int cl = 0; int pt = 0;
            char pname[MIDI_PORT_NAME_MAX];
            if (!midi_alsa_port_at_(c, st.is_input, (int)index, &cl, &pt, &pname[0], (unsigned long)MIDI_PORT_NAME_MAX))
                return audio_fail_(&st.error, AUDIO_ERR_INVALID_DEVICE, "alsa: no MIDI port with that index");
            int rc = midi_alsa_make_port_(st, c, name);
            if (rc != AUDIO_OK) return rc;
            int e = 0;
            unsafe { e = st.is_input ? g_aseq_.connect_from(c.seq, c.port, cl, pt) : g_aseq_.connect_to(c.seq, c.port, cl, pt); }
            if (e < 0) {
                midi_alsa_drop_port_(c);
                return audio_fail2_(&st.error, AUDIO_ERR_SYSTEM, "alsa: could not connect to", &pname[0]);
            }
            rc = midi_alsa_start_input_(st, c);
            if (rc != AUDIO_OK) midi_alsa_drop_port_(c);
            return rc;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static int midi_alsa_open_virtual_(&MidiPortState_ st, const &char name) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            int rc = midi_alsa_make_port_(st, c, name);
            if (rc != AUDIO_OK) return rc;
            rc = midi_alsa_start_input_(st, c);
            if (rc != AUDIO_OK) midi_alsa_drop_port_(c);
            return rc;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_alsa_close_port_(&MidiPortState_ st) {
    match (midi_alsa_of_(st)) {
        case some(c): { midi_alsa_drop_port_(c); }
        case null: {}
    }
}

static int midi_alsa_send_(&MidiPortState_ st, const &unsigned char msg, unsigned long len) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            if (c.port < 0) return audio_fail_(&st.error, AUDIO_ERR_INVALID_USE, "alsa: no port open");
            unsafe {
                if (len > c.coder_size) {
                    if (g_aseq_.mev_resize(c.coder, (size_t)len) < 0)
                        return audio_fail_(&st.error, AUDIO_ERR_NO_MEMORY, "alsa: message too long");
                    c.coder_size = len;
                }
                g_aseq_.mev_reset_encode(c.coder);
                unsigned long done = 0UL;
                while (done < len) {
                    struct AlsaSeqEvent_ ev;
                    safe_memset_raw_((void*)&ev, 0, sizeof(struct AlsaSeqEvent_));
                    long used = g_aseq_.mev_encode(c.coder, (const unsigned char*)msg + done, (long)(len - done), &ev);
                    if (used <= 0L) return audio_fail_(&st.error, AUDIO_ERR_INVALID_PARAM, "alsa: could not encode the message");
                    done = done + (unsigned long)used;
                    if (ev.type == (unsigned char)0) continue;   // incomplete so far (SND_SEQ_EVENT_NONE)
                    ev.source_port = (unsigned char)c.port;
                    ev.dest_client = (unsigned char)ASEQ_ADDRESS_SUBSCRIBERS;
                    ev.dest_port = (unsigned char)ASEQ_ADDRESS_UNKNOWN;
                    ev.queue = (unsigned char)ASEQ_QUEUE_DIRECT;
                    int r = g_aseq_.event_output_direct(c.seq, &ev);
                    if (r < 0) return audio_fail_code_(&st.error, AUDIO_ERR_SYSTEM, "alsa: snd_seq_event_output_direct", (long long)r);
                }
            }
            return AUDIO_OK;
        }
        case null: {}
    }
    return AUDIO_ERR_INVALID_USE;
}

static void midi_alsa_client_close_(&MidiPortState_ st) {
    match (midi_alsa_of_(st)) {
        case some(c): {
            midi_alsa_drop_port_(c);
            unsafe {
                if (c.coder != (void*)0) g_aseq_.mev_free(c.coder);
                g_aseq_.close(c.seq);
                dealloc_raw_((void*)c.decode_buf);
                dealloc_raw_((void*)c);
                st.backend = (void*)0;
            }
        }
        case null: {}
    }
}

static struct MidiBackendVt_ g_midi_alsa_vt_;
static int g_midi_alsa_vt_ready_ = 0;

&static MidiBackendVt_ midi_alsa_backend_() {
    if (!g_midi_alsa_vt_ready_) {
        g_midi_alsa_vt_.api = MIDI_API_ALSA;
        g_midi_alsa_vt_.available = midi_alsa_available_;
        g_midi_alsa_vt_.client_open = midi_alsa_client_open_;
        g_midi_alsa_vt_.port_count = midi_alsa_port_count_;
        g_midi_alsa_vt_.port_name = midi_alsa_port_name_;
        g_midi_alsa_vt_.open_port = midi_alsa_open_port_;
        g_midi_alsa_vt_.open_virtual = midi_alsa_open_virtual_;
        g_midi_alsa_vt_.close_port = midi_alsa_close_port_;
        g_midi_alsa_vt_.send = midi_alsa_send_;
        g_midi_alsa_vt_.client_close = midi_alsa_client_close_;
        g_midi_alsa_vt_ready_ = 1;
    }
    return &g_midi_alsa_vt_;
}

} // namespace std
