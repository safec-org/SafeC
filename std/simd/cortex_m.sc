// SafeC Standard Library — std::simd, ARM Cortex-M DSP extension (impl)
#pragma once
#include <std/simd/cortex_m.h>

namespace std {

inline int dsp_qadd(int a, int b)   { return __arm_dsp_qadd(a, b); }
inline int dsp_qsub(int a, int b)   { return __arm_dsp_qsub(a, b); }
inline int dsp_qadd16(int a, int b) { return __arm_dsp_qadd16(a, b); }
inline int dsp_qadd8(int a, int b)  { return __arm_dsp_qadd8(a, b); }
inline int dsp_qsub16(int a, int b) { return __arm_dsp_qsub16(a, b); }
inline int dsp_qsub8(int a, int b)  { return __arm_dsp_qsub8(a, b); }
inline int dsp_sadd16(int a, int b) { return __arm_dsp_sadd16(a, b); }
inline int dsp_sadd8(int a, int b)  { return __arm_dsp_sadd8(a, b); }
inline int dsp_ssub16(int a, int b) { return __arm_dsp_ssub16(a, b); }
inline int dsp_ssub8(int a, int b)  { return __arm_dsp_ssub8(a, b); }

inline unsigned int dsp_uqadd16(unsigned int a, unsigned int b) { return __arm_dsp_uqadd16(a, b); }
inline unsigned int dsp_uqadd8(unsigned int a, unsigned int b)  { return __arm_dsp_uqadd8(a, b); }
inline unsigned int dsp_uqsub16(unsigned int a, unsigned int b) { return __arm_dsp_uqsub16(a, b); }
inline unsigned int dsp_uqsub8(unsigned int a, unsigned int b)  { return __arm_dsp_uqsub8(a, b); }

inline int dsp_smlad(int a, int b, int acc)  { return __arm_dsp_smlad(a, b, acc); }
inline int dsp_smladx(int a, int b, int acc) { return __arm_dsp_smladx(a, b, acc); }
inline int dsp_smlsd(int a, int b, int acc)  { return __arm_dsp_smlsd(a, b, acc); }
inline int dsp_smlsdx(int a, int b, int acc) { return __arm_dsp_smlsdx(a, b, acc); }
inline int dsp_smuad(int a, int b)  { return __arm_dsp_smuad(a, b); }
inline int dsp_smuadx(int a, int b) { return __arm_dsp_smuadx(a, b); }
inline int dsp_smusd(int a, int b)  { return __arm_dsp_smusd(a, b); }
inline int dsp_smusdx(int a, int b) { return __arm_dsp_smusdx(a, b); }

inline unsigned int dsp_usad8(unsigned int a, unsigned int b) { return __arm_dsp_usad8(a, b); }
inline unsigned int dsp_usada8(unsigned int a, unsigned int b, unsigned int acc) {
    return __arm_dsp_usada8(a, b, acc);
}

inline int dsp_sxtab16(int a, int b) { return __arm_dsp_sxtab16(a, b); }
inline unsigned int dsp_uxtab16(unsigned int a, unsigned int b) { return __arm_dsp_uxtab16(a, b); }

} // namespace std
