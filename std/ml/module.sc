// SafeC Standard Library — models as composable traits (see module.h).
#pragma once
#include <std/ml/module.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

&FloatTensor recurrent_row_(const &FloatTensor xs, unsigned long t) {
    &FloatTensor idx = FloatTensor::new_1d(1UL, 0);
    unsafe { idx->data[0] = (float)t; }
    return tensor_gather_rows(xs, idx);
}

} // namespace std
