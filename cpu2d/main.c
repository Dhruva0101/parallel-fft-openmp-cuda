/* main.c: registers the CPU 2D implementations with the shared harness.
 * Author: Om Kumar Singh */
#include "harness.h"
#include "fft2d.h"

int main(int argc, char **argv) {
    const fft_impl impls[] = {
        {"seq", fft2d_seq},
        {"omp", fft2d_omp},
    };
    return harness_main(argc, argv, /*dim=*/2, impls, 2);
}
