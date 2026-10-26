#pragma once
// Internal: included by the Metal backend sources (gpu_mps.sc,
// device_mps.sc), not a public header. The opaque types of the Objective-C
// runtime and libdispatch they call Metal through besides gpu_mps.h's
// MtlObject (Objective-C's 'id': every MTLDevice, MTLBuffer, encoder,
// NSString, ...): a selector, an autorelease pool token. None is ever constructed or read;
// only the runtime looks behind them. Each objc_msgSend call site casts
// objc_msgSend to the typed signature of the message it sends.

#include <std/ml/gpu_mps.h>

namespace std {

struct ObjcSel_ { char opaque_; };         // SEL
struct ObjcPool_ { char opaque_; };        // an objc_autoreleasePoolPush token
struct DispatchQueue_ { char opaque_; };   // dispatch_queue_t (std passes none)
struct DispatchBlock_ { char opaque_; };   // dispatch_block_t (std passes none)

} // namespace std
