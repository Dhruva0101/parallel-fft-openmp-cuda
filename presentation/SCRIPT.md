# Talk script: 4:00 + 0:30 Q&A

25 clicks. **[click]** = advance. About 600 words: this is tight, so speak
while the animations play and don't wait for them to finish.

Speaker split:
**Dhruva** (0:00 – 1:45, clicks 1–12): Title, Why, Questions, Method, all of 1D.
**Om** (1:45 – 4:00, clicks 13 to the end): all of 2D, scaling, all six, wrap-up and Q&A lead.

## WHY (0:00 – 0:30) · Dhruva

**Title.** "We're Dhruva and Om. We built six parallel versions of the
Cooley–Tukey Fast Fourier Transform, on the CPU with OpenMP and on the GPU with CUDA."
**[click]**

**Why the FFT.** "An FFT turns a signal into its frequency components. It's
used everywhere, from audio to MRI to deep learning. Cooley–Tukey cuts the
work from N squared to N log N. It's also a good test for parallel hardware:
it has two kinds of parallelism, it's limited by memory rather than math, and
every answer can be checked." **[click]**

**Questions.** "We ask: how well does the CPU scale with more cores, when does
the GPU win, and does 1D versus 2D shape matter?" **[click]**

## METHOD (0:30 – 0:45) · Dhruva

**Fair test.** "All six get the same seeded input and the same timer. Small
sizes are checked against a direct DFT, and every size up to 16.8 million
values passes an inverse round trip. We ran on NC State's ARC cluster."
**[click]**

**Reading the charts.** "Speedup is sequential time over parallel time.
OpenMP uses 16 threads, one per core; the scaling test varies that from 1
to 16. The GPU gets two numbers: kernel only, and end to end with transfers.
1D and 2D use the same sizes, so all six line up." **[click]**

## 1D (0:45 – 1:45) · Dhruva

**1D methods** *(3 clicks, one per column)*.
Sequential: "One long FFT, using iterative radix-2 Cooley–Tukey. The basic
step is a butterfly. We reorder by bit-reversed index, then run log-N stages
one after another: the baseline." **[click]**
OpenMP: "Inside a stage, butterflies are independent, so we split them evenly
across threads, with a barrier after every stage." **[click]**
CUDA: "One GPU thread per butterfly. Blocks can only sync internally,
so the first 11 stages run fused in shared memory, and every later stage gets
its own launch, which acts as a grid-wide barrier." **[click]**

**1D compared** *(1:15)*. "Speedup over one core as the problem grows."
**[click]** "OpenMP at 1,000 values: 0.34×, slower: threads cost more than
the work." **[click]** "At 4 million: 8×. Big problems hide the overhead."
**[click]** "The GPU at 16.8 million: 38× end to end, 72× kernel only. The
kernel-only number leaves out transfers and host overhead. Om will take
us through 2D." **[click]** *(hand over to Om)*

## 2D (1:45 – 2:55) · Om

**2D methods** *(3 clicks, one per column)*.
Sequential: "Now an n-by-n grid: FFT every row, then every column,
copying each column into a buffer because columns are far apart in memory."
**[click]**
OpenMP: "Threads take whole rows, one barrier, then whole columns: one
barrier in total instead of log-N." **[click]**
CUDA: "One block per row in shared memory; columns are slow on a GPU, so we
transpose, reuse the row kernel, and transpose back." **[click]**

**2D compared** *(2:20)*. "Speedup over one core." **[click]** "OpenMP 2D is
already faster at 1,000 values and reaches 16×. The GPU beats the sequential
code from 4,000 values, and at 16.8 million reaches 90× end to end, 875×
kernel only." **[click]** "The gap is everything outside the kernels: 90 to
95 percent of GPU time." **[click]** "5 ms in kernels, 42 ms in transfers and
overhead." **[click]**

## ALL SIX (2:55 – 4:00) · Om

**Q1 scaling** *(2:55)*. "Now fix the size and add cores. Each line is
speedup over the same code on one core; the grey curve is ideal." **[click]**
"2D at 16.8 million: 14.1× on 16 cores, 88 percent of ideal, because with one
barrier more cores just means more rows at once. 1D at the same size only
reaches 8×, and 3 to 5× at smaller sizes: its log-N barriers cost more as
cores are added." **[click]**

**All six**. "All six on one chart, lower is faster; same number of
values, different data layout." **[click]** "Shape matters once you go
parallel: at 16.8 million, OpenMP 2D is 1.4× faster than 1D, and up to 3× at
smaller sizes. On the GPU, the 2D kernels are 9.6× faster, because short rows
stay in shared memory." **[click]**

**What we found.** "To sum up: 1D scales 8× and 2D 14× on 16 cores, CUDA 2D
is 90× end to end, and 2D beats 1D. So parallelism pays off when the problem
is big enough to hide the overhead, the work needs few barriers, and the data
stays where the compute is." **[click]** "Thank you, questions?"

---

## Likely questions (30 s)

- **Why measure scaling against 1 core of the same code?** That isolates
  what extra cores add. At 2²⁴ on ARC, one-thread OpenMP 1D is about 1.1×
  faster than the sequential code (3.07 s vs 3.34 s), so the two baselines
  are close but not identical.
- **Why is sequential 2D slower than sequential 1D?** Same number of
  butterflies, but 2D also gathers every column into a buffer and scatters it
  back: 1.1–1.4× slower at every size.
- **Why is CUDA 2D compute faster than CUDA 1D?** 2D keeps each row in shared
  memory for all its stages; 1D's later stages each read and write global
  memory once.
- **Is comparing 1D and 2D fair?** Same number of values = same number of
  butterflies (N/2 · log₂N); only the data layout and parallel structure
  differ.
- **Would pinned memory reduce the transfer time?** Probably, partly: at 16.8 M
  values we move 268 MB in about 42 ms, roughly 6 GB/s from pageable memory,
  well below what PCIe 4.0 ×8 allows.
