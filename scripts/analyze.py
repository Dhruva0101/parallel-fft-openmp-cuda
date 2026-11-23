#!/usr/bin/env python3
"""
analyze.py: turn results/*.csv into a summary table and plots.

    python3 scripts/analyze.py            # needs: pip install pandas matplotlib

Method (docs/BENCHMARKING.md):
  * run 0 of every configuration is a warm-up and is DROPPED;
  * remaining runs -> mean and standard deviation;
  * speedup S = T_serial / T_parallel, where T_serial is the sequential
    implementation of the SAME dimension at the SAME size;
  * CUDA gets two series: end-to-end (total_ms, includes PCIe copies) and
    kernel-only (kernel_ms). The gap between them is a key finding.

Outputs:
  results/summary.md                 Markdown tables (also the accessible table view)
  plots/runtime_{1d,2d}.png          runtime vs. element count (log-log)
  plots/speedup_{1d,2d}.png          speedup vs. element count
"""
from pathlib import Path
import sys

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("analyze.py needs pandas and matplotlib:  pip install pandas matplotlib")

ROOT = Path(__file__).resolve().parent.parent
RESULTS, PLOTS = ROOT / "results", ROOT / "plots"

# Each implementation keeps the same colour and marker in every plot.
# Markers and end-of-line labels make the series readable without colour.
SERIES = {
    "seq":         dict(label="Sequential",        color="#2a78d6", marker="o"),
    "omp":         dict(label="OpenMP",            color="#eb6834", marker="s"),
    "cuda":        dict(label="CUDA (end-to-end)", color="#1baf7a", marker="^"),
    "cuda_kernel": dict(label="CUDA (kernel only)", color="#eda100", marker="D"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def load() -> pd.DataFrame:
    frames = [pd.read_csv(RESULTS / f"{name}.csv")
          for name in ("cpu1d", "cpu2d", "cuda1d", "cuda2d")
          if (RESULTS / f"{name}.csv").exists()]
    if not frames:
        sys.exit("no results/*.csv yet - run `make bench` first")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["run"] > 0]                                    # drop warm-up

    # Turn CUDA kernel_ms into its own series so it's plotted like the others.
    k = df[df["kernel_ms"] >= 0].copy()
    k["impl"], k["total_ms"] = "cuda_kernel", k["kernel_ms"]
    df = pd.concat([df, k], ignore_index=True)

    g = (df.groupby(["dim", "impl", "size", "elements"])["total_ms"]
           .agg(mean_ms="mean", std_ms="std").reset_index())
    base = (g[g["impl"] == "seq"][["dim", "size", "mean_ms"]]
              .rename(columns={"mean_ms": "seq_ms"}))
    g = g.merge(base, on=["dim", "size"], how="left")
    g["speedup"] = g["seq_ms"] / g["mean_ms"]
    return g


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Elements (N for 1D, n×n for 2D)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot(g, dim, metric, fname, title, ylabel, logy):
    d = g[g["dim"] == dim]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for impl, spec in SERIES.items():
        s = d[d["impl"] == impl].sort_values("elements")
        if s.empty or (metric == "speedup" and impl == "seq"):
            continue
        ax.plot(s["elements"], s[metric], color=spec["color"], marker=spec["marker"],
                markersize=6, linewidth=2, label=spec["label"],
                markeredgecolor="white", markeredgewidth=1)
        last = s.iloc[-1]                                     # direct label at line end
        ax.annotate(spec["label"], (last["elements"], last[metric]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    if metric == "speedup":
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("break-even (1×)", xy=(1.0, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=MUTED)
    if logy:
        ax.set_yscale("log")
    style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/{fname}")


def summary(g):
    lines = ["# Results summary", "",
             "Mean of timed runs (warm-up dropped). Speedup = T_seq / T_impl.", ""]
    for dim in sorted(g["dim"].unique()):
        d = g[g["dim"] == dim]
        lines += [f"## {dim}D FFT", "",
                  "| size | elements | implementation | mean (ms) | std (ms) | speedup |",
                  "|---:|---:|---|---:|---:|---:|"]
        for _, r in d.sort_values(["elements", "impl"]).iterrows():
            sp = "" if pd.isna(r["speedup"]) else f"{r['speedup']:.2f}×"
            lines.append(f"| {r['size']} | {r['elements']} | {SERIES[r['impl']]['label']} "
                         f"| {r['mean_ms']:.3f} | {r['std_ms']:.3f} | {sp} |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("wrote results/summary.md")


if __name__ == "__main__":
    g = load()
    summary(g)
    for dim in (1, 2):
        plot(g, dim, "mean_ms", f"runtime_{dim}d.png", f"{dim}D FFT runtime",
             "Time (ms, log scale)", logy=True)
        plot(g, dim, "speedup", f"speedup_{dim}d.png", f"{dim}D FFT speedup over sequential",
             "Speedup (×)", logy=False)
