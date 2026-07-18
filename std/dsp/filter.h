// SafeC Standard Library — General FIR / IIR Filters
//
// Streaming (one-sample-at-a-time) difference-equation filters — the
// general form biquad.h's 2nd-order sections and dsp.h's array-level
// primitives both specialize:
//
//   FIR ("feedforward"): y[n] = sum_{k=0}^{M-1} b[k]*x[n-k]
//   IIR ("feedback"):     y[n] = sum_{k=0}^{M-1} b[k]*x[n-k]
//                              - sum_{k=1}^{N-1} a[k]*y[n-k]      (a[0] = 1)
//
// Both keep their own history/state internally (a delay line for FIR, a
// direct-form-I input+output history for IIR) so a caller can feed them
// one real-time sample at a time — e.g. inside std::Reactor's per-task
// processing — rather than needing the whole signal in memory up front
// the way convolution.h's conv_direct does.
#pragma once
#include <std/dsp/fixed.h>

namespace std {

// ── FIR ──────────────────────────────────────────────────────────────────────

struct FirFilter {
    const double* coeffs;    // b[0..numTaps-1], caller-owned, must outlive the filter
    unsigned long numTaps;
    double*       history;   // caller-provided ring buffer, numTaps elements, zero-initialized
    unsigned long pos;       // ring buffer write position

    // Processes one sample, returns y[n].
    double process(double x);

    // Resets the history to all zeros (numTaps elements).
    void   reset();
};

// coeffs/history as in struct FirFilter — 'history' must have numTaps
// elements (caller-allocated, typically zero-initialized).
struct FirFilter fir_init(const double* coeffs, unsigned long numTaps, double* history);

// Processes 'n' samples through 'f', writing to 'out' (may alias 'in').
void fir_process_block(struct FirFilter* f, const double* in, double* out, unsigned long n);

struct FFirFilter {
    const Fixed*  coeffs;
    unsigned long numTaps;
    Fixed*        history;
    unsigned long pos;

    Fixed process(Fixed x);
    void  reset();
};

struct FFirFilter ffir_init(const Fixed* coeffs, unsigned long numTaps, Fixed* history);
void ffir_process_block(struct FFirFilter* f, const Fixed* in, Fixed* out, unsigned long n);

// ── IIR (general order, direct form I) ────────────────────────────────────────

struct IirFilter {
    const double* b;         // feedforward coeffs, b[0..numB-1]
    unsigned long numB;
    const double* a;         // feedback coeffs, a[1..numA-1] used (a[0] assumed 1 — pass your a[] with a[0]=1 anyway for clarity)
    unsigned long numA;
    double*       xHistory;  // caller-provided, numB elements, zero-initialized
    double*       yHistory;  // caller-provided, numA elements, zero-initialized
    unsigned long xPos;
    unsigned long yPos;

    double process(double x);
    void   reset();
};

struct IirFilter iir_init(const double* b, unsigned long numB,
                           const double* a, unsigned long numA,
                           double* xHistory, double* yHistory);
void iir_process_block(struct IirFilter* f, const double* in, double* out, unsigned long n);

struct FIirFilter {
    const Fixed*  b;
    unsigned long numB;
    const Fixed*  a;
    unsigned long numA;
    Fixed*        xHistory;
    Fixed*        yHistory;
    unsigned long xPos;
    unsigned long yPos;

    Fixed process(Fixed x);
    void  reset();
};

struct FIirFilter fiir_init(const Fixed* b, unsigned long numB,
                             const Fixed* a, unsigned long numA,
                             Fixed* xHistory, Fixed* yHistory);
void fiir_process_block(struct FIirFilter* f, const Fixed* in, Fixed* out, unsigned long n);

} // namespace std
