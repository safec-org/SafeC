// SafeC Standard Library — models as composable traits (see module.h).
#pragma once
#include <std/ml/module.h>
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

&Tensor recurrent_row_(const &Tensor xs, unsigned long t) {
    &Tensor idx = Tensor::new_1d(1UL, 0);
    unsafe { idx->data[0] = (float)t; }
    return tensor_gather_rows(xs, idx);
}

} // namespace std
