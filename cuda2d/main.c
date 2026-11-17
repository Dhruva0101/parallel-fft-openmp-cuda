/* main.c: registers the CUDA 2D implementation with the shared harness.
 * Author: Dhruva Kamble */
#include "harness.h"

double fft2d_cuda(cpx *x, int n);   /* defined in fft2d_cuda.cu (extern "C") */

int main(int argc, char **argv) {
    const fft_impl impls[] = {
        {"cuda", fft2d_cuda},
    };
    return harness_main(argc, argv, /*dim=*/2, impls, 1);
}
