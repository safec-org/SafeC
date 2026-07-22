#pragma once
// SafeC Standard Library — safetensors weight file reader.
//
// Format (github.com/huggingface/safetensors, a stable, simple spec):
//   [8 bytes] little-endian u64: N, the byte length of the header
//   [N bytes] UTF-8 JSON: {"tensor.name": {"dtype":"BF16","shape":[..],
//             "data_offsets":[start,end]}, ..., "__metadata__": {...}}
//   [rest]    raw tensor bytes, back-to-back; a given tensor's bytes are at
//             absolute file offset (8 + N + start) .. (8 + N + end).
// "__metadata__" (if present) is a plain string->string map, not a tensor
// entry — skipped when building the tensor index.
//
// Real checkpoints (this module was written against google/gemma-4-E2B's
// single-file 10GB+ model.safetensors) store weights as BF16, not F32 — see
// std/ml/float16.h's bf16_to_f32_bulk, used here to convert on load since
// SafeC's Tensor (std/ml/tensor.h) is F32-only. F32-dtype tensors pass
// through with a plain byte copy (safetensors is little-endian throughout,
// matching every target this compiler runs on -- no byte-swap needed).
#include <std/ml/tensor.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

struct SafetensorsEntry {
    struct String name;
    struct String dtype;          // "BF16", "F32", "F16", ... (as written in the header)
    unsigned long shape[8];        // safetensors tensors are rank <= 8 in every
                                     // real checkpoint this targets; ndim below
                                     // says how many of these 8 slots are used
    unsigned long ndim;
    unsigned long dataOffsetStart; // relative to the start of the data section,
    unsigned long dataOffsetEnd;    // i.e. NOT counting the 8-byte length prefix
                                      // or the header JSON itself
};

struct SafetensorsFile {
    void*         handle;          // std::file_open handle, kept open for lazy loads
    unsigned long dataBaseOffset;  // absolute file offset where tensor data begins
    struct Vec    entries;         // Vec<struct SafetensorsEntry>

    // Index into 'entries' by name, or -1 if not present. Linear scan --
    // real checkpoints have a few hundred entries, this runs once per
    // weight at model-load time, not in any hot path.
    long          find(const char* name) const;

    // Loads the named tensor into a freshly-allocated F32 Tensor (converting
    // from BF16 if that's how it's stored), shape taken directly from the
    // header. Returns (Tensor*)0 (with a printf'd error) if the name isn't
    // found or the dtype isn't one this function knows how to widen to F32.
    // Caller owns the result (free with Tensor::free() when done).
    struct Tensor* load(const char* name);

    void close_();
};

// Opens 'path', reads and parses the header, and builds the name -> entry
// index. Does NOT read any tensor data yet (that's what .load() is for) --
// safe to call on a 10GB+ file without touching most of it. On failure
// (file not found, bad magic/header), returns a SafetensorsFile with
// handle == (void*)0; check that before calling any other method on it.
struct SafetensorsFile safetensors_open(const char* path);

} // namespace std
