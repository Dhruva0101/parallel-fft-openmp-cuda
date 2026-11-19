# Results summary

Mean of timed runs (warm-up dropped). Speedup = T_seq / T_impl.

## 1D FFT

| size | elements | implementation | mean (ms) | std (ms) | speedup |
|---:|---:|---|---:|---:|---:|
| 1024 | 1024 | OpenMP | 0.036 | 0.001 | 0.45× |
| 1024 | 1024 | Sequential | 0.016 | 0.000 | 1.00× |
| 4096 | 4096 | OpenMP | 0.062 | 0.001 | 1.23× |
| 4096 | 4096 | Sequential | 0.076 | 0.000 | 1.00× |
| 16384 | 16384 | OpenMP | 0.150 | 0.014 | 2.41× |
| 16384 | 16384 | Sequential | 0.362 | 0.001 | 1.00× |
| 65536 | 65536 | OpenMP | 0.492 | 0.010 | 3.62× |
| 65536 | 65536 | Sequential | 1.781 | 0.008 | 1.00× |
| 262144 | 262144 | OpenMP | 1.905 | 0.054 | 4.76× |
| 262144 | 262144 | Sequential | 9.060 | 0.010 | 1.00× |
| 1048576 | 1048576 | OpenMP | 8.502 | 0.151 | 6.45× |
| 1048576 | 1048576 | Sequential | 54.811 | 0.312 | 1.00× |
| 4194304 | 4194304 | OpenMP | 44.876 | 0.320 | 13.98× |
| 4194304 | 4194304 | Sequential | 627.398 | 0.787 | 1.00× |
| 16777216 | 16777216 | OpenMP | 417.389 | 0.635 | 12.51× |
| 16777216 | 16777216 | Sequential | 5221.937 | 3.875 | 1.00× |

## 2D FFT

| size | elements | implementation | mean (ms) | std (ms) | speedup |
|---:|---:|---|---:|---:|---:|
| 32 | 1024 | CUDA (end-to-end) | 0.021 | 0.001 |  |
| 32 | 1024 | CUDA (kernel only) | 0.009 | 0.001 |  |
| 64 | 4096 | CUDA (end-to-end) | 0.027 | 0.001 |  |
| 64 | 4096 | CUDA (kernel only) | 0.008 | 0.000 |  |
| 128 | 16384 | CUDA (end-to-end) | 0.053 | 0.002 |  |
| 128 | 16384 | CUDA (kernel only) | 0.013 | 0.001 |  |
| 256 | 65536 | CUDA (end-to-end) | 0.145 | 0.004 |  |
| 256 | 65536 | CUDA (kernel only) | 0.019 | 0.003 |  |
| 512 | 262144 | CUDA (end-to-end) | 0.423 | 0.002 |  |
| 512 | 262144 | CUDA (kernel only) | 0.044 | 0.002 |  |
| 1024 | 1048576 | CUDA (end-to-end) | 1.501 | 0.008 |  |
| 1024 | 1048576 | CUDA (kernel only) | 0.154 | 0.003 |  |
| 2048 | 4194304 | CUDA (end-to-end) | 6.360 | 0.004 |  |
| 2048 | 4194304 | CUDA (kernel only) | 1.175 | 0.003 |  |
| 4096 | 16777216 | CUDA (end-to-end) | 25.522 | 0.005 |  |
| 4096 | 16777216 | CUDA (kernel only) | 4.987 | 0.004 |  |
