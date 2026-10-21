#pragma once
// SafeC Standard Library — capability traits shared across modules.
//
// A trait that more than one module's roles are built from lives here, so
// each module includes it instead of declaring its own copy.
//
//   Resettable  reset(): back to the initial state. An arena-style
//               allocator frees every block at once (alloc/allocator.h's
//               ArenaAllocator); a filter clears its history, so the next
//               output is the response from silence (dsp/sample_filter.h's
//               SampleFilter).

namespace std {

trait Resettable {
    void reset();
}

} // namespace std
