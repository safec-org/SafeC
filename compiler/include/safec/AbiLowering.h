#pragma once

namespace llvm { class Module; }

namespace safec {

// Rewrites large aggregate returns into the target C ABI's sret form and
// large aggregate load/store copies into memcpy. Run once per module after
// CodeGen::generate() has emitted every body (see AbiLowering.cpp).
void lowerAggregateABI(llvm::Module &m);

} // namespace safec
