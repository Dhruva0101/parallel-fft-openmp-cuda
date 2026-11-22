"""
talk.py: 4-minute class presentation, built with Manim + manim-slides.

Flow:  WHY (why the FFT, research questions)
       HOW (how we measure; 1D: three methods on one slide + comparison; same for 2D)
       WHAT (all six compared, takeaways)

Render, then export (from the presentation/ folder). Delete slides/ and media/
first: manim-slides reuses cached clips across quality settings.
    rm -rf slides media
    .venv/bin/manim-slides render -qk --fps 30 talk.py FFTTalk
    .venv/bin/manim-slides convert FFTTalk talk.html --one-file --offline

Charts read results/*.csv, so re-render after new benchmark runs.
"""
import csv
import os
from collections import defaultdict

from manim import *
from manim_slides import Slide

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# ---- palette: NC State brand colours ---------------------------------------
RED = "#CC0000"      # Wolfpack Red (primary)
BG = "#FFFFFF"
FG = "#000000"
MUTED = "#5E5E5E"
DIM = "#D9D9D9"
SEQ = "#6F6F6F"      # sequential
OMP = RED            # OpenMP / CPU
GPU = "#427E93"      # CUDA: Innovation Blue
ACC = "#4156A1"      # Bio-Indigo
BAD = "#990000"      # Reynolds Red
GOOD = "#6F7D1C"     # Genomic Green
THREADS = ["#4156A1", "#D14905", "#6F7D1C", "#008473"]  # Indigo, Pyroman Flame, Genomic Green, Aqua
FONT = "Helvetica Neue"   # NC State's Univers is not installed; closest available
FS = 0.85                 # global font scale

config.background_color = BG


def T(s, size=30, color=FG, **kw):
    return Text(s, font=FONT, font_size=size * FS, color=color, **kw)


def M(s, size=30, color=FG, **kw):
    return MarkupText(s, font=FONT, font_size=size * FS, color=color, **kw)


def red_bar():
    return Rectangle(width=config.frame_width, height=0.14, fill_color=RED, fill_opacity=1,
                     stroke_width=0).to_edge(UP, buff=0)


def fit(m, w):
    """Shrink m to width w if it is wider (never enlarges)."""
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def right_col(m, top=2.3, left=0.7, width=6.0):
    """Place m as the right-hand text column: left edge fixed, width capped."""
    fit(m, width)
    m.move_to([left + m.width / 2, top - m.height / 2, 0])
    return m


def caption(s, size=28):
    return fit(M(s, size), 13.2).to_edge(DOWN, buff=0.4)


# ---- data ------------------------------------------------------------------
def load(name):
    """Mean total_ms / kernel_ms per (impl, size), warm-up run dropped."""
    path = os.path.join(RESULTS, name)
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["run"] == "0":
                continue
            a = acc[(r["impl"], int(r["size"]))]
            a[0] += float(r["total_ms"])
            a[1] += float(r["kernel_ms"])
            a[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


CPU1D = load("cpu1d.csv")
CUDA1D = load("cuda1d.csv")
CPU2D = load("cpu2d.csv")
CUDA2D = load("cuda2d.csv")


def series(data, impl, dim, kernel=False):
    """{element count: mean ms}. 2D sizes are matrix sides, so elements = n*n."""
    out = {}
    for (i, size), (tot, ker) in data.items():
        if i == impl:
            out[size if dim == 1 else size * size] = ker if kernel else tot
    return out


S1 = {"seq": series(CPU1D, "seq", 1), "omp": series(CPU1D, "omp", 1),
      "cuda": series(CUDA1D, "cuda", 1), "cudak": series(CUDA1D, "cuda", 1, kernel=True)}
S2 = {"seq": series(CPU2D, "seq", 2), "omp": series(CPU2D, "omp", 2),
      "cuda": series(CUDA2D, "cuda", 2), "cudak": series(CUDA2D, "cuda", 2, kernel=True)}


def load_scaling():
    """{(dim, elements, threads): mean OpenMP ms} from results/scaling.csv."""
    path = os.path.join(RESULTS, "scaling.csv")
    acc = defaultdict(list)
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["run"] != "0" and r["impl"] == "omp":
                    acc[(int(r["dim"]), int(r["elements"]), int(r["threads"]))].append(float(r["total_ms"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


SCALE = load_scaling()


def speedups(base, d):
    return {e: base[e] / v for e, v in d.items() if e in base and v > 0}

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow2(k):
    return "2" + str(k).translate(SUP)


# ---- slide chrome ----------------------------------------------------------
IMPLS = [  # (dimension, name, colour)
    ("1D", "Sequential", SEQ), ("1D", "OpenMP", OMP), ("1D", "CUDA", GPU),
    ("2D", "Sequential", SEQ), ("2D", "OpenMP", OMP), ("2D", "CUDA", GPU),
]


def crumb(active):
    parts = VGroup()
    names = ["WHY", "METHOD", "1D", "2D", "ALL SIX"]
    for i, name in enumerate(names):
        on = name == active
        parts.add(T(name, 22, RED if on else MUTED, weight=BOLD if on else NORMAL))
        if i < len(names) - 1:
            parts.add(T("›", 24, MUTED))
    parts.arrange(RIGHT, buff=0.2).to_corner(UR, buff=0.35)
    return parts


def minimap(active):
    """2 x 3 grid of the six implementations, current one filled."""
    cells = VGroup()
    for k, (_, _, col) in enumerate(IMPLS):
        sq = Square(0.26, stroke_color=col, stroke_width=2,
                    fill_color=col, fill_opacity=1.0 if k in active else 0.0)
        cells.add(sq)
    cells.arrange_in_grid(rows=2, cols=3, buff=0.08)
    lbl = VGroup(T("1D", 18, MUTED), T("2D", 18, MUTED)).arrange(DOWN, buff=0.15)
    lbl.next_to(cells, LEFT, buff=0.12)
    return VGroup(lbl, cells)


def header(title, active, impl=None, color=FG):
    c = crumb(active)
    t = fit(T(title, 40, color, weight=BOLD), 12.6 - c.width).to_corner(UL, buff=0.45)
    g = VGroup(red_bar(), t, c)
    if impl is not None:
        g.add(minimap(impl).next_to(c, DOWN, buff=0.2).align_to(c, RIGHT))
    return g


# ---- charts ----------------------------------------------------------------
def make_axes(ylo, yhi, ystep, ylabels, xtitle, log, xlen=7.2, ylen=4.4):
    """x: log2(values) 10..24. y: value (linear) or log10(value), shifted so the
    x-axis sits at the bottom. Returns (axes, decorations, y-mapping function)."""
    span = yhi - ylo
    ax = Axes(x_range=[10, 24, 2], y_range=[0, span, ystep], x_length=xlen, y_length=ylen,
              axis_config={"color": MUTED, "include_tip": False, "stroke_width": 2.5})
    ax.move_to([-6.6 + 1.0 + xlen / 2, -0.35, 0])

    def ymap(v):
        y = (np.log10(v) if log else v) - ylo
        return float(np.clip(y, 0, span))

    xl = VGroup(*[T(pow2(k), 24, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.18) for k in range(10, 25, 2)])
    yl = VGroup(*[T(lbl, 24, MUTED).next_to(ax.c2p(10, ymap(v)), LEFT, buff=0.15) for v, lbl in ylabels])
    xt = T(xtitle, 24, MUTED).next_to(xl, DOWN, buff=0.2)
    return ax, VGroup(xl, yl, xt), ymap


def hline(ax, ymap, v, color, label, width=2, opacity=1.0, left=False):
    ln = DashedLine(ax.c2p(10, ymap(v)), ax.c2p(24, ymap(v)), color=color, dash_length=0.08,
                    stroke_width=width, stroke_opacity=opacity)
    t = T(label, 20, color)
    if left:
        t.next_to(ax.c2p(10, ymap(v)), UR, buff=0.06)
    else:
        t.next_to(ax.c2p(24, ymap(v)), UP, buff=0.06).shift(LEFT * 0.6)
    return VGroup(ln, t)


def plot(ax, ymap, d, color, dashed=False):
    pts = [(np.log2(e), ymap(v)) for e, v in sorted(d.items())]
    line = VMobject(stroke_color=color, stroke_width=5).set_points_as_corners(
        [ax.c2p(x, y) for x, y in pts])
    if dashed:
        line = DashedVMobject(line, num_dashes=40, dashed_ratio=0.6)
    dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.065, color=color) for x, y in pts])
    return VGroup(line, dots)


def legend_entry(label, color, dashed, pending):
    sw = (DashedLine if dashed else Line)(ORIGIN, RIGHT * 0.6, stroke_width=5,
                                          color=MUTED if pending else color)
    txt = T(label + ("  (pending)" if pending else ""), 24, MUTED if pending else FG)
    return VGroup(sw, txt).arrange(RIGHT, buff=0.2)


# ---- butterfly diagram -----------------------------------------------------
class Butterfly:
    """Radix-2 DIT butterfly network; butterflies use the flat numbering of
    fft1d_omp.c / the CUDA stage kernel:  b -> (i, i + half)."""

    def __init__(self, N, left, right, top, gap, color=FG, width=3):
        self.N, self.S = N, int(np.log2(N))
        self.xs = np.linspace(left, right, self.S + 1)
        self.ys = [top - gap * r for r in range(N)]
        self.dots = VGroup(*[Dot(self.p(c, r), radius=0.06, color=FG)
                             for c in range(self.S + 1) for r in range(N)])
        self.bfl = {}
        for s in range(1, self.S + 1):
            half = 1 << (s - 1)
            self.bfl[s] = []
            for b in range(N // 2):
                i = ((b >> (s - 1)) << s) + (b & (half - 1))
                g = VGroup(
                    Line(self.p(s - 1, i), self.p(s, i)),
                    Line(self.p(s - 1, i), self.p(s, i + half)),
                    Line(self.p(s - 1, i + half), self.p(s, i)),
                    Line(self.p(s - 1, i + half), self.p(s, i + half)),
                ).set_stroke(color, width)
                self.bfl[s].append(g)

    def p(self, c, r):
        return np.array([self.xs[c], self.ys[r], 0])

    def stage(self, s):
        return VGroup(*self.bfl[s])

    def stage_labels(self, size=24):
        return VGroup(*[T(f"stage {s}", size, MUTED).move_to(
            (self.p(s - 1, self.N - 1) + self.p(s, self.N - 1)) / 2 + DOWN * 0.5)
            for s in range(1, self.S + 1)])


# ---- matrix grid -----------------------------------------------------------
def grid(n, side, center, color=SEQ):
    cells = {}
    g = VGroup()
    for r in range(n):
        for c in range(n):
            sq = Square(side * 0.88, stroke_width=0, fill_color=color, fill_opacity=0.85)
            sq.move_to(center + np.array([(c - (n - 1) / 2) * side, -(r - (n - 1) / 2) * side, 0]))
            cells[(r, c)] = sq
            g.add(sq)
    return g, cells


def row_of(cells, n, r):
    return VGroup(*[cells[(r, c)] for c in range(n)])


def col_of(cells, n, c):
    return VGroup(*[cells[(r, c)] for r in range(n)])


class FFTTalk(Slide):
    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)

    def construct(self):
        self.title_slide()
        self.why_slide()
        self.method_slide()
        self.methods1d_slide()
        self.compare1d_slide()
        self.methods2d_slide()
        self.compare2d_slide()
        self.scaling_slide()
        self.compare_all_slide()
        self.takeaways_slide()

    # ------------------------------------------------------------------ WHY
    def title_slide(self):
        q = T("Parallel FFT: OpenMP vs CUDA", 60, weight=BOLD)
        sub = T("Six implementations of the Fast Fourier Transform, on the CPU and the GPU", 32)
        who = T("Dhruva Kamble · Om Kumar Singh", 28, MUTED)
        course = T("CSC 548 Parallel Systems · NC State · Fall 2026", 24, MUTED)
        VGroup(q, sub, who, course).arrange(DOWN, buff=0.35)
        who.shift(DOWN * 0.4); course.shift(DOWN * 0.4)
        rule = Line(LEFT * 2.5, RIGHT * 2.5, color=RED, stroke_width=4).next_to(sub, DOWN, buff=0.35)
        self.add(red_bar())
        self.play(Write(q), run_time=1.0)
        self.play(Create(rule), FadeIn(sub, shift=UP * 0.2), FadeIn(who), FadeIn(course))
        self.next_slide()

    def why_slide(self):
        self.clear_all()
        h = header("Why the FFT?", "WHY")
        self.play(FadeIn(h))
        uses = M(f'Used everywhere: audio · images · MRI · wireless · deep learning.   '
                 f'N² work → <span foreground="{RED}">N log₂N</span>', 30).shift(UP * 2.0)
        fit(uses, 12.8)
        lead = T("And a good test for parallel hardware:", 30, weight=BOLD).next_to(uses, DOWN, buff=0.6)

        def reason(title, body):
            t = T(title, 27, RED, weight=BOLD)
            b = T(body, 25, FG, line_spacing=0.9)
            g = VGroup(t, b).arrange(DOWN, buff=0.25)
            box = RoundedRectangle(width=4.1, height=2.3, corner_radius=0.12,
                                   stroke_color=DIM, stroke_width=2)
            g.move_to(box)
            return VGroup(box, g)

        reasons = VGroup(
            reason("Two kinds of parallelism", "inside one long FFT (1D)\nvs. across many rows (2D)"),
            reason("Limited by memory", "little math per value:\ndata movement sets speed"),
            reason("Known right answer", "every version can be\nchecked exactly"),
        ).arrange(RIGHT, buff=0.3).next_to(lead, DOWN, buff=0.45)
        self.play(FadeIn(uses))
        self.play(FadeIn(lead), LaggedStart(*[FadeIn(r, shift=UP * 0.2) for r in reasons], lag_ratio=0.2))
        self.next_slide()

        self.play(FadeOut(uses), FadeOut(lead), FadeOut(reasons), run_time=0.4)
        box_q = fit(T("When does CPU or GPU parallelism pay off for the FFT?", 34, weight=BOLD), 12.2)
        box = SurroundingRectangle(box_q, color=RED, buff=0.3, corner_radius=0.1)
        rq = VGroup(box, box_q).shift(UP * 1.2)
        qs = VGroup(
            M(f'<span foreground="{OMP}"><b>Q1</b></span>  How well does the CPU scale with more cores?', 32),
            M(f'<span foreground="{GPU}"><b>Q2</b></span>  When does the GPU win, and what limits it?', 32),
            M(f'<span foreground="{ACC}"><b>Q3</b></span>  Does the shape matter: one long 1D FFT vs. a 2D grid?', 32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(rq, DOWN, buff=0.6)
        self.play(Create(box), Write(box_q), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(q, shift=RIGHT * 0.3) for q in qs], lag_ratio=0.2))
        self.next_slide()

    # ------------------------------------------------------------------ HOW
    COLS = (-4.45, 0.0, 4.45)      # centres of the three method columns
    COLW = 4.0

    def method_frame(self, title, dim):
        """Header plus three empty column titles and dividers. Returns titles."""
        h = header(title, dim, impl=[0, 1, 2] if dim == "1D" else [3, 4, 5])
        divs = VGroup(*[Line([x, 2.6, 0], [x, -3.5, 0], color=DIM, stroke_width=1.5) for x in (-2.22, 2.22)])
        names = [("Sequential", SEQ), ("OpenMP", OMP), ("CUDA", GPU)]
        titles = VGroup(*[T(f"{n} {dim}", 30, c, weight=BOLD).move_to([x, 2.3, 0])
                          for (n, c), x in zip(names, self.COLS)])
        self.play(FadeIn(h), FadeIn(divs))
        return titles

    def method_text(self, x, lines):
        """lines: (text, colour) pairs, placed under the column's diagram."""
        g = VGroup(*[T(s, 23, c) for s, c in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        fit(g, self.COLW - 0.2)
        return g.move_to([x, -1.95, 0]).align_to(np.array([x - (self.COLW - 0.2) / 2, 0, 0]), LEFT)

    def methods1d_slide(self):
        self.clear_all()
        titles = self.method_frame("1D: three ways to run one long FFT", "1D")
        L, W, TOP, GAP = 1.45, 2.9, 1.5, 0.27

        # Sequential: every stage in turn, one core
        x = self.COLS[0]
        bf = Butterfly(8, x - L, x + L, TOP, GAP, color=SEQ, width=2.5)
        txt = self.method_text(x, [("Reorder by bit-reversed index,", FG),
                                   ("then log₂N stages of butterflies,", FG),
                                   ("one after another on 1 core", FG),
                                   ("→ the 1D baseline", MUTED)])
        self.play(FadeIn(titles[0]), FadeIn(bf.dots))
        for s in (1, 2, 3):
            self.play(LaggedStart(*[Create(g) for g in bf.bfl[s]], lag_ratio=0.2), run_time=0.45)
        self.play(FadeIn(txt))
        self.next_slide()

        # OpenMP: each stage split across threads, barrier between stages
        x = self.COLS[1]
        bf = Butterfly(8, x - L, x + L, TOP, GAP, color=SEQ, width=2.5)
        bars = VGroup(*[DashedLine(bf.p(c, 0) + UP * 0.2, bf.p(c, 7) + DOWN * 0.05, color=FG, dash_length=0.08)
                        for c in (1, 2)])
        leg = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.35, color=THREADS[t], stroke_width=6), T(f"thread {t}", 20))
                       .arrange(RIGHT, buff=0.12) for t in (0, 1)]).arrange(RIGHT, buff=0.3).next_to(bf.dots, DOWN, buff=0.28)
        txt = self.method_text(x, [("Split each stage's butterflies", FG),
                                   ("evenly across threads", FG),
                                   ("Barrier after every stage", FG),
                                   ("→ log₂N barriers", MUTED)])
        self.play(FadeIn(titles[1]), FadeIn(bf.dots), *[Create(g) for s in bf.bfl for g in bf.bfl[s]])
        self.play(*[g.animate.set_stroke(THREADS[0 if b < 2 else 1], 3.5)
                    for s in bf.bfl for b, g in enumerate(bf.bfl[s])], FadeIn(leg))
        self.play(Create(bars), FadeIn(txt))
        self.next_slide()

        # CUDA: one thread per butterfly; blocks fuse early stages, launches order later ones
        x = self.COLS[2]
        bf = Butterfly(8, x - L, x + L, TOP, GAP, color=GPU, width=2.2)
        bands = VGroup()
        for k in range(2):
            top, bot = bf.ys[4 * k] + 0.1, bf.ys[4 * k + 3] - 0.1
            r = Rectangle(width=bf.xs[2] - bf.xs[0] + 0.3, height=top - bot, stroke_color=GPU,
                          stroke_width=1.5, fill_color=GPU, fill_opacity=0.1)
            bands.add(r.move_to([(bf.xs[0] + bf.xs[2]) / 2, (top + bot) / 2, 0]))
        bl = VGroup(*[T(f"block {k}", 18, GPU).rotate(PI / 2).next_to(bands[k], LEFT, buff=0.06) for k in range(2)])
        k1 = T("1 launch, fused", 18, GPU).next_to(bands[0], UP, buff=0.06)
        k2 = T("launch 2", 18, FG).move_to([(bf.xs[2] + bf.xs[3]) / 2, k1.get_y(), 0])
        txt = self.method_text(x, [("1 GPU thread per butterfly", FG),
                                   ("Blocks only sync internally, so", FG),
                                   ("later stages = 1 launch each", FG),
                                   ("→ first 11 stages fused in", MUTED),
                                   ("   fast shared memory", MUTED)])
        self.play(FadeIn(titles[2]), FadeIn(bf.dots), *[Create(g) for s in bf.bfl for g in bf.bfl[s]])
        self.play(FadeIn(bands), FadeIn(bl), FadeIn(k1), FadeIn(k2))
        self.play(Indicate(bf.stage(3), color=BAD, scale_factor=1.02), FadeIn(txt))
        self.next_slide()

    def methods2d_slide(self):
        self.clear_all()
        titles = self.method_frame("2D: three ways to run an n × n grid", "2D")
        n, side, CY = 6, 0.36, 0.75

        def rows_pass(cells, run_time=0.8):
            boxes = [SurroundingRectangle(row_of(cells, n, r), color=FG, buff=0.02, stroke_width=3) for r in range(n)]
            self.play(*[ShowPassingFlash(b, time_width=0.8) for b in boxes], run_time=run_time)

        # Sequential: row by row, then column by column through a buffer
        x = self.COLS[0]
        g, cells = grid(n, side, np.array([x, CY, 0]))
        txt = self.method_text(x, [("FFT every row, then every column", FG),
                                   ("Columns are far apart in memory,", FG),
                                   ("so each is copied to a buffer", FG),
                                   ("→ the 2D baseline", MUTED)])
        self.play(FadeIn(titles[0]), FadeIn(g))
        hl = SurroundingRectangle(row_of(cells, n, 0), color=FG, buff=0.03)
        self.play(Create(hl), run_time=0.2)
        for r in range(1, n):
            self.play(hl.animate.move_to(row_of(cells, n, r)), run_time=0.1)
        self.play(Transform(hl, SurroundingRectangle(col_of(cells, n, 0), color=ACC, buff=0.03)), run_time=0.25)
        for c in range(1, n):
            self.play(hl.animate.move_to(col_of(cells, n, c)), run_time=0.1)
        self.play(FadeOut(hl), FadeIn(txt), run_time=0.3)
        self.next_slide()

        # OpenMP: whole rows per thread, one barrier, whole columns per thread
        x = self.COLS[1]
        g, cells = grid(n, side, np.array([x, CY, 0]))
        leg = VGroup(*[VGroup(Square(0.2, fill_color=THREADS[t], fill_opacity=1, stroke_width=0), T(f"thread {t}", 20))
                       .arrange(RIGHT, buff=0.1) for t in range(3)]).arrange(RIGHT, buff=0.22).next_to(g, DOWN, buff=0.15)
        txt = self.method_text(x, [("Threads take whole rows,", FG),
                                   ("1 barrier, then whole columns", FG),
                                   ("Each thread has its own buffer", FG),
                                   ("→ 1 barrier instead of log₂N", MUTED)])
        self.play(FadeIn(titles[1]), FadeIn(g), FadeIn(leg))
        self.play(*[cells[(r, c)].animate.set_fill(THREADS[r // 2]) for r in range(n) for c in range(n)])
        self.play(*[cells[(r, c)].animate.set_fill(THREADS[c // 2]) for r in range(n) for c in range(n)])
        self.play(FadeIn(txt))
        self.next_slide()

        # CUDA: rows in shared memory, transpose, rows again, transpose back
        x = self.COLS[2]
        cols = color_gradient(["#4156A1", "#427E93", "#6F7D1C", "#D14905", RED], n)
        g, cells = grid(n, side, np.array([x, CY, 0]))
        for (r, c), sq in cells.items():
            sq.set_fill(cols[c])
        pos = {k: v.get_center() for k, v in cells.items()}
        step = T("rows", 20, MUTED).next_to(g, DOWN, buff=0.15)
        txt = self.method_text(x, [("1 thread block per row,", FG),
                                   ("row held in shared memory", FG),
                                   ("Transpose instead of reading", FG),
                                   ("columns (32 × 32 tiles)", FG)])
        self.play(FadeIn(titles[2]), FadeIn(g), FadeIn(step))

        def transpose(label):
            self.play(Transform(step, T(label, 20, MUTED).move_to(step)), run_time=0.2)
            new = {(c, r): sq for (r, c), sq in cells.items()}
            self.play(*[sq.animate.move_to(pos[k]) for k, sq in new.items()], run_time=0.8)
            cells.clear(); cells.update(new)

        rows_pass(cells)
        transpose("transpose")
        self.play(Transform(step, T("rows again", 20, MUTED).move_to(step)), run_time=0.2)
        rows_pass(cells)
        transpose("transpose back")
        self.play(FadeIn(txt))
        self.next_slide()

    # --------------------------------------------------- how we measure
    def method_slide(self):
        self.clear_all()
        h = header("How we measure and show results", "METHOD")
        self.play(FadeIn(h))

        def card(title, body, color):
            t = T(title, 28, color, weight=BOLD)
            b = VGroup(*[T(line, 24) for line in body]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            content = VGroup(t, b).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            box = RoundedRectangle(width=4.1, height=2.45, corner_radius=0.12,
                                   stroke_color=color, stroke_width=2.5)
            content.move_to(box)
            return VGroup(box, content)

        cards = VGroup(
            card("Same for all six", ["same random input", "same compiler flags",
                                      "same timer, 5 timed runs"], ACC),
            card("Checked correct", ["vs. an exact DFT", "and IFFT(FFT(x)) = x,",
                                     "up to 16.8 M values"], GOOD),
            card("Real hardware", ["NCSU ARC cluster", "CPU: 16-core AMD EPYC",
                                   "GPU: NVIDIA RTX 4060 Ti"], RED),
        ).arrange(RIGHT, buff=0.3).shift(UP * 1.0)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.2))
        self.next_slide()

        title = T("How to read the charts", 30, weight=BOLD).next_to(cards, DOWN, buff=0.55).align_to(cards, LEFT)
        items = VGroup(
            M('<b>Speedup</b> = sequential time ÷ parallel time.  Above 1× means faster.', 26),
            M('<b>Two GPU numbers</b>: compute only, and including copies to and from the GPU.', 26),
            M('<b>Same sizes for 1D and 2D</b>: 2¹⁰ to 2²⁴ values (2D uses n × n), so all six line up.', 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        fit(items, 12.8)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in items], lag_ratio=0.2))
        self.next_slide()

    # ------------------------------------------------------------------ WHAT
    def speedup_chart(self, base, entries, dim):
        """entries: (label, data, colour, dashed). Linear axis while every speedup
        is <= 40x, log axis once CUDA speedups make it larger."""
        sps = [(e, speedups(base, e[1])) for e in entries]
        top = max([max(sp.values()) for _, sp in sps if sp], default=1)
        if top <= 40:
            yhi = 16 if top <= 16 else 8 * int(np.ceil(top / 8))
            ax, deco, ymap = make_axes(0, yhi, yhi / 4, [(v, f"{v:g}×") for v in np.linspace(0, yhi, 5)],
                                       f"values ({'N' if dim == 1 else 'n × n'}), speedup over sequential {dim}D",
                                       log=False)
        else:
            ax, deco, ymap = make_axes(-1, 3, 1, [(0.1, "0.1×"), (1, "1×"), (10, "10×"), (100, "100×"), (1000, "1000×")],
                                       f"values ({'N' if dim == 1 else 'n × n'}), speedup over sequential {dim}D",
                                       log=True)
        even = hline(ax, ymap, 1, MUTED, "break-even")
        lines, legend = [], VGroup()
        for (label, d, col, dashed), sp in sps:
            legend.add(legend_entry(label, col, dashed, pending=not sp))
            if sp:
                lines.append(plot(ax, ymap, sp, col, dashed))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        right_col(legend, top=2.4, left=2.0, width=4.6)
        self.play(Create(ax), FadeIn(deco), FadeIn(even), FadeIn(legend))
        if not base:
            self.play(FadeIn(T(f"waiting for the sequential {dim}D baseline (Om)", 26, MUTED).move_to(ax)))
        for ln in lines:
            self.play(Create(ln[0]), FadeIn(ln[1]), run_time=1.0)
        return ax, ymap, lines, legend

    def note(self, parts, anchor, gap=0.4):
        """parts: list of (text, colour). First line is the coloured lead."""
        g = VGroup(*[T(s, 24, c) for s, c in parts]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        return fit(g, 4.6).next_to(anchor, DOWN, buff=gap, aligned_edge=LEFT)

    def compare1d_slide(self):
        self.clear_all()
        h = header("1D compared: speedup as the problem grows", "1D", color=OMP)
        self.play(FadeIn(h))
        ax, ymap, lines, legend = self.speedup_chart(S1["seq"], [
            ("OpenMP 1D", S1["omp"], OMP, False),
            ("CUDA 1D, compute only", S1["cudak"], GPU, True),
            ("CUDA 1D, incl. copies", S1["cuda"], GPU, False),
        ], 1)
        self.play(FadeIn(hline(ax, ymap, 16, OMP, "16 cores", opacity=0.7, left=True)))
        self.next_slide()

        sp = speedups(S1["seq"], S1["omp"])
        if not sp:
            return
        e0 = min(sp); eb = max(sp, key=sp.get); ks = sorted(sp)
        pts = lines[0][1]
        a1 = self.note([(f"OpenMP: {sp[e0]:.2f}× at 1K values", OMP),
                        ("threads cost more than the work", FG)], legend, 0.45)
        self.play(Indicate(pts[ks.index(e0)], color=FG, scale_factor=2), FadeIn(a1))
        self.next_slide()
        a2 = self.note([(f"{sp[eb]:.0f}× at {pow2(int(np.log2(eb)))} values", OMP),
                        ("big problems hide the overhead", FG)], a1)
        self.play(Indicate(pts[ks.index(eb)], color=FG, scale_factor=2), FadeIn(a2))
        self.next_slide()
        ce, ck = speedups(S1["seq"], S1["cuda"]), speedups(S1["seq"], S1["cudak"])
        if ce:
            top = max(ce)
            a3 = self.note([(f"GPU at {pow2(int(np.log2(top)))}: {ce[top]:.0f}× incl. copies,", GPU),
                            (f"{ck[top]:.0f}× compute only", GPU),
                            ("dip at 2²²: 32 MB outgrows L2", FG)], a2)
            self.play(FadeIn(a3))
            self.next_slide()

    def compare2d_slide(self):
        self.clear_all()
        h = header("2D compared · Q2: when does the GPU win?", "2D", color=GPU)
        self.play(FadeIn(h))
        ax, ymap, lines, legend = self.speedup_chart(S2["seq"], [
            ("OpenMP 2D", S2["omp"], OMP, False),
            ("CUDA 2D, compute only", S2["cudak"], GPU, True),
            ("CUDA 2D, incl. copies", S2["cuda"], GPU, False),
        ], 2)
        self.play(FadeIn(hline(ax, ymap, 16, OMP, "16 cores", opacity=0.7, left=True)))
        self.next_slide()

        so, ce, ck = (speedups(S2["seq"], S2[k]) for k in ("omp", "cuda", "cudak"))
        if so and ce:
            lo, top = min(so), max(ce)
            b1 = self.note([(f"OpenMP 2D: {so[lo]:.1f}× already at 1K values", OMP),
                            (f"up to {max(so.values()):.0f}×: whole rows, 1 barrier", FG)], legend, 0.45)
            win = min(e for e, v in ce.items() if v > 1)
            b2 = self.note([(f"GPU wins from {pow2(int(np.log2(win)))} values", GPU),
                            (f"at {pow2(int(np.log2(top)))}: {ce[top]:.0f}× incl. copies,", FG),
                            (f"{ck[top]:.0f}× compute only", FG)], b1)
            self.play(FadeIn(b1))
            self.play(FadeIn(b2))
            self.next_slide()

        # where does the GPU time go?
        self.play(*[FadeOut(m) for m in self.mobjects if m is not h], run_time=0.4)
        sub = T("Why the gap between the two GPU lines: copies", 30, weight=BOLD).next_to(h, DOWN, buff=0.35).to_edge(LEFT, buff=0.6)
        W = 7.6
        rows = VGroup()
        for n in (256, 1024, 4096):
            total, kern = CUDA2D[("cuda", n)]
            kf = kern / total
            k = Rectangle(width=W * kf, height=0.7, fill_color=GPU, fill_opacity=1, stroke_width=0)
            c = Rectangle(width=W * (1 - kf), height=0.7, fill_color=DIM, fill_opacity=1, stroke_width=0)
            bar = VGroup(k, c).arrange(RIGHT, buff=0)
            rows.add(VGroup(T(f"{n} × {n}", 26), bar, T(f"copying {100 * (1 - kf):.0f}%", 26),
                            T(f"{total:.2f} ms", 26, MUTED)))
        for i, r in enumerate(rows):
            r[1].move_to(UP * (0.9 - 0.95 * i) + RIGHT * 0.4)
            r[0].next_to(r[1], LEFT, buff=0.3)
            r[2].move_to(r[1][1])
            r[3].next_to(r[1], RIGHT, buff=0.3)
        legend = VGroup(
            VGroup(Square(0.26, fill_color=GPU, fill_opacity=1, stroke_width=0), T("GPU compute", 24)).arrange(RIGHT),
            VGroup(Square(0.26, fill_color=DIM, fill_opacity=1, stroke_width=0), T("copy to/from GPU + overhead", 24)).arrange(RIGHT),
        ).arrange(RIGHT, buff=0.6).next_to(rows, UP, buff=0.4)
        self.play(FadeIn(sub), FadeIn(legend))
        for r in rows:
            self.play(GrowFromEdge(r[1], LEFT), FadeIn(r[0]), FadeIn(r[2]), FadeIn(r[3]), run_time=0.5)
        t, k = CUDA2D[("cuda", 4096)]
        msg = VGroup(
            M(f'16.8 M values: the FFT takes <span foreground="{GPU}">{k:.0f} ms</span>, '
              f'moving the data takes <span foreground="{MUTED}">{t - k:.0f} ms</span>', 28),
            M('<b>The GPU wins big only if the data already lives on the GPU.</b>', 30),
        ).arrange(DOWN, buff=0.25).to_edge(DOWN, buff=0.55)
        self.next_slide()
        self.play(FadeIn(msg, shift=UP * 0.2))
        self.next_slide()

    def scaling_slide(self):
        """Strong scaling: OpenMP speedup over the same code on 1 thread."""
        if not SCALE:
            return
        self.clear_all()
        h = header("Q1 · How does the CPU scale with cores?", "ALL SIX", color=OMP)
        self.play(FadeIn(h))
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 16, 4], x_length=7.2, y_length=4.4,
                  axis_config={"color": MUTED, "include_tip": False, "stroke_width": 2.5})
        ax.move_to([-6.6 + 1.0 + 3.6, -0.35, 0])
        xl = VGroup(*[T(str(2 ** k), 24, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.18) for k in range(5)])
        yl = VGroup(*[T(f"{v}×", 24, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in range(0, 17, 4)])
        xt = T("cores (OpenMP threads)", 24, MUTED).next_to(xl, DOWN, buff=0.2)
        yt = T("speedup over the same code on 1 core", 22, MUTED).next_to(ax, UP, buff=0.15).align_to(ax, LEFT)
        # x = log2(cores), so ideal speedup (= cores) is the curve through (k, 2^k)
        ideal = DashedVMobject(VMobject(stroke_color=MUTED, stroke_width=3).set_points_as_corners(
            [ax.c2p(k, 2 ** k) for k in range(5)]), num_dashes=30, dashed_ratio=0.5)
        ideal_l = T("ideal = cores", 20, MUTED).next_to(ax.c2p(3, 8), UL, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), Create(ideal), FadeIn(ideal_l))

        shades = {1: ["#F4A6A6", "#E05252", "#990000"], 2: ["#A9B4DD", "#6A7BC4", "#2E3F86"]}
        ends = []
        for dim in (2, 1):
            for i, e in enumerate(sorted({k[1] for k in SCALE if k[0] == dim})):
                base = SCALE[(dim, e, 1)]
                pts = [(np.log2(t), base / SCALE[(dim, e, t)]) for t in (1, 2, 4, 8, 16)]
                col = shades[dim][i]
                line = VMobject(stroke_color=col, stroke_width=5).set_points_as_corners([ax.c2p(x, y) for x, y in pts])
                if dim == 1:
                    line = DashedVMobject(line, num_dashes=30, dashed_ratio=0.6)
                dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.06, color=col) for x, y in pts])
                lbl = T(f"{dim}D {pow2(int(np.log2(e)))}", 20, col).next_to(ax.c2p(4, pts[-1][1]), RIGHT, buff=0.12)
                ends.append((dim, e, pts[-1][1]))
                self.play(Create(line), FadeIn(dots), FadeIn(lbl), run_time=0.6)
        self.next_slide()

        best2 = max(v for d, e, v in ends if d == 2)
        r1 = [v for d, e, v in ends if d == 1]
        notes = VGroup(
            T(f"2D: {best2:.1f}× on 16 cores", 26, "#2E3F86", weight=BOLD),
            T(f"{100 * best2 / 16:.0f}% of ideal", 24),
            T("1 barrier: more cores,", 24),
            T("more rows at once", 24),
            T(f"1D: only {min(r1):.0f}–{max(r1):.0f}×", 26, RED, weight=BOLD),
            T("log₂N barriers,", 24),
            T("worst when small", 24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        for m in notes[4:]:
            m.shift(DOWN * 0.35)
        right_col(notes, top=1.2, left=2.9, width=3.9)
        self.play(FadeIn(notes))
        self.next_slide()

    def shape_ratios(self):
        """OpenMP 2D vs 1D (total time) and CUDA 2D vs 1D (compute only), same sizes."""
        o = [S1["omp"][e] / S2["omp"][e] for e in S1["omp"] if e in S2["omp"]]
        g = [S1["cudak"][e] / S2["cudak"][e] for e in S1["cudak"] if e in S2["cudak"]]
        return o, g

    def compare_all_slide(self):
        self.clear_all()
        h = header("All six · Q3: does the shape matter?", "ALL SIX", color=ACC)
        self.play(FadeIn(h))
        ax, deco, ymap = make_axes(-2, 4, 1, [(0.01, "0.01 ms"), (1, "1 ms"), (100, "100 ms"), (10000, "10 s")],
                                   "values (N for 1D, n × n for 2D): lower is faster", log=True)
        entries = [
            ("Sequential 1D", S1["seq"], SEQ, False), ("OpenMP 1D", S1["omp"], OMP, False),
            ("CUDA 1D", S1["cuda"], GPU, False),
            ("Sequential 2D", S2["seq"], SEQ, True), ("OpenMP 2D", S2["omp"], OMP, True),
            ("CUDA 2D", S2["cuda"], GPU, True),
        ]
        legend = VGroup(*[legend_entry(l, c, dsh, pending=not d) for l, d, c, dsh in entries]
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        key = VGroup(T("solid = 1D, dashed = 2D", 22, MUTED),
                     T("GPU times include copies", 22, MUTED),
                     T("same values = same arithmetic", 22, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        right_col(VGroup(legend, key).arrange(DOWN, aligned_edge=LEFT, buff=0.25), top=2.4, left=2.0, width=4.6)
        self.play(Create(ax), FadeIn(deco), FadeIn(legend), FadeIn(key))
        for l, d, c, dsh in entries:
            if d:
                ln = plot(ax, ymap, d, c, dsh)
                self.play(Create(ln[0]), FadeIn(ln[1]), run_time=0.7)
        self.next_slide()

        o, g = self.shape_ratios()
        if o and g:
            f = self.note([("Yes: 2D beats 1D once parallel", ACC),
                           (f"OpenMP 2D: {min(o):.1f}–{max(o):.1f}× faster than 1D", FG),
                           ("  (1 barrier instead of log₂N)", MUTED),
                           (f"GPU compute: 2D up to {max(g):.0f}× faster", FG),
                           ("  (rows stay in shared memory)", MUTED)], key, 0.35)
            self.play(FadeIn(f))
        self.next_slide()

    def takeaways_slide(self):
        self.clear_all()
        h = header("What we found", "ALL SIX")
        self.play(FadeIn(h))
        sp = speedups(S1["seq"], S1["omp"])
        best = max(sp.values()) if sp else 0
        ce = speedups(S2["seq"], S2["cuda"])
        o, g = self.shape_ratios()
        q1 = f'Pays off above ~4K values; up to {best:.0f}× on 16 cores'
        if SCALE:
            s2 = max(SCALE[(2, e, 1)] / SCALE[(2, e, 16)] for (d, e, t) in SCALE if d == 2 and t == 16)
            s1 = max(SCALE[(1, e, 1)] / SCALE[(1, e, 16)] for (d, e, t) in SCALE if d == 1 and t == 16)
            q1 = f'2D scales almost perfectly ({s2:.1f}× on 16 cores); 1D only up to {s1:.0f}×'
        q2 = (f'Wins from a few thousand values, up to {max(ce.values()):.0f}× incl. copies; '
              f'copies are most of its time') if ce else 'Compute is fast, but copies dominate'
        q3 = (f'2D beats 1D in parallel: OpenMP {min(o):.1f}–{max(o):.1f}×, '
              f'GPU compute up to {max(g):.0f}×') if o and g else '[ 1D vs. 2D result ]'
        items = VGroup(
            M(f'<span foreground="{OMP}"><b>Q1 CPU</b></span>   {q1}', 30),
            M(f'<span foreground="{GPU}"><b>Q2 GPU</b></span>   {q2}', 30),
            M(f'<span foreground="{ACC}"><b>Q3 Shape</b></span>   {q3}', 30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(UP * 0.7)
        fit(items, 13.0)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in items], lag_ratio=0.3))
        ans = VGroup(T("Parallelism pays off when the problem is big enough to hide the overhead,", 30, weight=BOLD),
                     T("the work needs few barriers, and the data stays where the compute is.", 30, weight=BOLD)
                     ).arrange(DOWN, buff=0.15).next_to(items, DOWN, buff=0.7)
        fit(ans, 13.0)
        self.play(FadeIn(ans))
        self.next_slide()
        thanks = T("Questions?", 56, RED, weight=BOLD).to_edge(DOWN, buff=0.5)
        self.play(ans.animate.shift(UP * 0.2), Write(thanks))
        self.next_slide()
