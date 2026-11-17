/* main.c: registers the CPU 1D implementations with the shared harness.
 * Author: Dhruva Kamble */
#include "harness.h"
#include "fft1d.h"

int main(int argc, char **argv) {
    const fft_impl impls[] = {
        {"seq", fft1d_seq},
        {"omp", fft1d_omp},
    };
    return harness_main(argc, argv, /*dim=*/1, impls, 2);
}
