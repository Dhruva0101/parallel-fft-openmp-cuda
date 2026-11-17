/* main.c: registers the CUDA 1D implementation with the shared harness.
 * Author: Om Kumar Singh */
#include "harness.h"

double fft1d_cuda(cpx *x, int n);   /* defined in fft1d_cuda.cu (extern "C") */

int main(int argc, char **argv) {
    const fft_impl impls[] = {
        {"cuda", fft1d_cuda},
    };
    return harness_main(argc, argv, /*dim=*/1, impls, 1);
}
