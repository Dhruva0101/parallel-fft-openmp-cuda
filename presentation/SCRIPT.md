# Talk script: 4:00 + 0:30 Q&A

25 clicks. **[click]** = advance. About 600 words: this is tight, so speak
while the animations play and don't wait for them to finish.

Suggested split: each person explains what they wrote.
**Dhruva**: Why, Method, Sequential/OpenMP 1D, 1D compared, CUDA 2D, 2D compared.
**Om**: CUDA 1D, Sequential/OpenMP 2D, all six / Q3. Either: wrap-up.

> CPU 2D and CUDA 1D numbers currently come from stand-in code. Re-run them
> with Om's implementation before presenting, or say so in the talk.

## WHY (0:00 – 0:30)

**Title.** "We're Dhruva and Om. We built six parallel versions of the
Cooley–Tukey Fast Fourier Transform, on the CPU with OpenMP and on the GPU with CUDA."
**[click]**

**Why the FFT.** "The FFT is used everywhere, from audio to MRI to deep
learning. Cooley–Tukey cuts the work from N squared to N log N. It's also a good test for parallel hardware: it has two kinds of
parallelism, it's limited by memory rather than math, and every answer can
be checked exactly." **[click]**

**Questions.** "We ask: how well does the CPU scale with more cores, when does
the GPU win, and does 1D versus 2D shape matter?" **[click]**

## METHOD (0:30 – 0:45)

**Fair test.** "All six get the same input, flags and timer, and every result
is checked against an exact DFT. We ran on NC State's ARC cluster." **[click]**

**Reading the charts.** "Speedup is sequential time over parallel time. The
GPU gets two numbers: compute only, and including copies. 1D and 2D use the
same sizes, so all six line up." **[click]**

## 1D (0:45 – 1:45)

**1D methods** *(3 clicks, one per column)*.
Sequential: "One long FFT, using iterative radix-2 Cooley–Tukey. The basic step is a butterfly. We reorder by
bit-reversed index, then run log-N stages one after another: the baseline."
**[click]**
OpenMP: "Inside a stage, butterflies are independent, so we split them evenly
across threads, with a barrier after every stage." **[click]**
CUDA *(Om)*: "One GPU thread per butterfly. Blocks can only sync internally,
so the first 11 stages run fused in shared memory, and every later stage gets
its own launch, which acts as a grid-wide barrier." **[click]**

**1D compared** *(1:15)*. "Speedup over one core as the problem grows."
**[click]** "OpenMP at 1,000 values: 0.45×, slower: threads cost more than
the work." **[click]** "At 4 million: 14×. Big problems hide the overhead."
**[click]** "The GPU: 91× including copies, 143× compute only; it dips where
the data outgrows its L2 cache." **[click]**

## 2D (1:45 – 2:55)

**2D methods** *(3 clicks, one per column)*.
Sequential *(Om)*: "Now an n-by-n grid: FFT every row, then every column,
copying each column into a buffer because columns are far apart in memory."
**[click]**
OpenMP *(Om)*: "Threads take whole rows, one barrier, then whole columns: one
barrier in total instead of log-N." **[click]**
CUDA: "One block per row in shared memory; columns are slow on a GPU, so we
transpose, reuse the row kernel, and transpose back." **[click]**

**2D compared** *(2:20)*. "Speedup over one core." **[click]** "OpenMP 2D is
already faster at 1,000 values and reaches 14×. The GPU reaches 117× including
copies, 600× compute only." **[click]** "The gap between those two is copying:
80 to 90 percent of GPU time." **[click]** "5 ms of compute, 21 of copying."
**[click]**

## ALL SIX (2:55 – 4:00)

**Q1 scaling** *(2:55)*. "Now fix the size and add cores. Each line is
speedup over the same code on one core; the grey curve is ideal." **[click]**
"2D sits almost on the ideal: 15.7× on 16 cores, because with one barrier
more cores just means more rows at once. 1D only reaches 4 to 8×: its
log-N barriers cost more as cores are added, worst when the problem is
small." **[click]**

**All six** *(Om)*. "All six on one chart, lower is faster; same values means
the same arithmetic." **[click]** "Shape matters once you go parallel: OpenMP
2D is 1.3 to 2.8× faster than 1D, because it needs one barrier instead of
log-N. On the GPU, 2D compute is up to 7× faster." **[click]**

**What we found.** "So parallelism pays off when the problem is big enough to
hide the overhead, the work needs few barriers, and the data stays where the
compute is." **[click]** "Thank you, questions?"

---

## Likely questions (30 s)

- **Why measure scaling against 1 core of the same code?** That isolates
  what extra cores add. On ARC at 2²⁴, one-thread OpenMP 1D is already 1.5×
  faster than the sequential code (not on a laptop); we haven't found why
  yet, so we keep that out of the scaling numbers.
- **Why is sequential 2D slower than 1D at small sizes but faster at large?**
  Small: gathering strided columns costs extra. Large: 2D works on short rows
  that fit in cache, while 1D's late stages jump across the whole array.
- **Why is CUDA 2D compute faster than CUDA 1D?** 2D keeps each row in shared
  memory for all its stages; 1D's later stages each read and write global
  memory once.
- **Is comparing 1D and 2D fair?** Same number of values = same number of
  butterflies (N/2 · log₂N); only the parallel structure differs.
- **Would pinned memory fix the copies?** Not much: we already move data at
  ~13 GB/s, close to the 4060 Ti's PCIe 4.0 ×8 limit.
