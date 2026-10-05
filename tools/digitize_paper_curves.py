#!/usr/bin/env python3
"""Digitize the source figure panels and overlay them on the replicated curves.

Reads the page renders (`pdftoppm -r 200`) of the published EUR curve figures,
extracts the blue FRA curves by pixel classification using calibrated plot
frames, writes `paper_*.csv` into the output directory and saves overlays for
the 1M, 3M and ON curves together with difference statistics. The statistics
are split into the year-end windows (turn-spanning samples) and the rest, for
the plain replica, the year-end overlay columns (`fwd1d_overlay`,
`fra1m_overlay`, `fra3m_overlay`) and the direct turn-knot columns
(`fra1m_direct`, `fra3m_direct_hyman`) dumped in the CSVs.
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Plot-area frames measured from the 200 dpi page renders: x_px spans the
# first and last labelled x tick (top panels) or the full x range endpoints
# (bottom panels), y_px are the pixel rows of two known y values.
FIGURES = {
    "fig30_top": dict(
        page="page-62.png", x_px=(367.5, 1400.0), x_data=(0.0, 2.0),
        y_px=(439.5, 915.0), y_data=(0.75, 0.0), loose=False,
        mask_boxes=((1080, 1400, 445, 590),),
    ),
    "fig30_bottom": dict(
        page="page-62.png", x_px=(369.5, 1404.5), x_data=(0.0, 60.0),
        y_px=(1127.0, 1601.0), y_data=(3.5, 0.0), loose=False,
        mask_boxes=((1160, 1410, 1540, 1605),),
    ),
    "fig26_top": dict(
        page="page-57.png", x_px=(375.0, 1397.0), x_data=(0.0, 2.0),
        y_px=(418.5, 742.5), y_data=(0.20, 0.0), loose=True,
        mask_boxes=((595, 1160, 812, 898),),
    ),
    "fig26_bottom": dict(
        page="page-57.png", x_px=(352.0, 1399.0), x_data=(0.0, 60.0),
        y_px=(1119.0, 1609.5), y_data=(3.5, 0.0), loose=True,
        mask_boxes=((580, 1140, 1512, 1602),),
    ),
    "fig28_top": dict(
        page="page-60.png", x_px=(368.0, 1405.5), x_data=(0.0, 2.0),
        y_px=(422.0, 889.0), y_data=(0.75, 0.0), loose=False,
        mask_boxes=((1165, 1400, 428, 500),),
    ),
}

# Named turn-spanning windows in years from the reference date, used to split
# the residuals into the discontinuity regions and the rest. A single turn
# produces a bump whose width is the FRA tenor, so the windows cover the
# straddling accruals and nothing more.
TURN_WINDOWS = {
    "fig26": [("end-2012", 0.00, 0.09), ("end-2013", 1.02, 1.09)],
    "fig28": [("end-2013", 0.94, 1.07)],
    "fig30": [("end-2013", 0.78, 1.07)],
}

# (figure, variant, mean|d|, max|d|) for the end-2013 windows, collected by
# report() and written as a cross-variant summary.
turn_summary = []


def blue_mask(img, loose):
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    if loose:
        return (b > r + 30) & (b >= g) & (b > 70)
    return (b > r + 40) & (b > g + 40) & (b > 90)


def digitize(pages_dir, cfg):
    img = np.asarray(Image.open(os.path.join(pages_dir, cfg["page"])).convert("RGB"))
    mask = blue_mask(img, cfg["loose"])
    for c0, c1, r0, r1 in cfg.get("mask_boxes", ()):
        mask[r0:r1 + 1, c0:c1 + 1] = False
    x0, x1 = cfg["x_px"]
    y0, y1 = cfg["y_px"]
    xa, xb = cfg["x_data"]
    ya, yb = cfg["y_data"]
    top, bottom = int(min(y0, y1)), int(max(y0, y1))
    samples = []
    for column in range(int(x0), int(x1) + 1):
        rows = np.nonzero(mask[top:bottom + 1, column])[0]
        if rows.size == 0:
            continue
        row = float(np.median(rows)) + top
        x = xa + (column - x0) / (x1 - x0) * (xb - xa)
        y = ya + (row - y0) / (y1 - y0) * (yb - ya)
        samples.append((x, y))
    return samples


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return np.array([float(row[name]) for row in rows])


def residual(paper_samples, our_x, our_y, limit):
    paper_x = np.array([s[0] for s in paper_samples])
    paper_y = np.array([s[1] for s in paper_samples])
    keep = paper_x <= limit
    paper_x, paper_y = paper_x[keep], paper_y[keep]
    ours = np.interp(paper_x, our_x, our_y)
    return paper_x, paper_y, ours, (ours - paper_y) * 100.0  # percent -> bp


def window_mask(t, begin, end):
    return (t >= begin) & (t <= end)


def describe(label, diff, t):
    if diff.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={diff.size} mean|d|={np.mean(np.abs(diff)):.2f} bp "
            f"max|d|={np.max(np.abs(diff)):.2f} bp "
            f"at t={t[np.argmax(np.abs(diff))]:.2f}y")


def report(name, paper_samples, our_x, our_y, limit, windows, title, lines,
           variant="plain", key=None):
    paper_x, paper_y, ours, diff = residual(paper_samples, our_x, our_y, limit)
    inside = np.zeros(paper_x.shape, dtype=bool)
    lines.append(f"  {title}")
    lines.append("    " + describe("all", diff, paper_x))
    for label, begin, end in windows:
        mask = window_mask(paper_x, begin, end)
        inside |= mask
        lines.append("    " + describe(label, diff[mask], paper_x[mask]))
        if key is not None and label == "end-2013" and diff[mask].size:
            turn_summary.append((key, variant, float(np.mean(np.abs(diff[mask]))),
                                 float(np.max(np.abs(diff[mask])))))
    lines.append("    " + describe("ex-year-end", diff[~inside], paper_x[~inside]))
    return paper_x, paper_y, ours, diff


def main():
    pages_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "internal_docs/paper_fit"
    os.makedirs(output, exist_ok=True)

    digits = {}
    for name, cfg in FIGURES.items():
        samples = digitize(pages_dir, cfg)
        digits[name] = samples
        with open(os.path.join(output, f"paper_{name}.csv"), "w", newline="") as handle:
            handle.write("t,rate_percent\n")
            for x, y in samples:
                handle.write(f"{x:.6f},{y:.6f}\n")

    fra = read_rows(os.path.join(output, "fra.csv"))
    ois = read_rows(os.path.join(output, "ois.csv"))
    our_fra_t = column(fra, "t")
    our_fra_1m = column(fra, "fra1m") * 100.0
    our_fra_1m_overlay = column(fra, "fra1m_overlay") * 100.0
    our_fra_1m_direct = column(fra, "fra1m_direct") * 100.0
    our_fra_3m = column(fra, "fra3m_hyman") * 100.0
    our_fra_3m_overlay = column(fra, "fra3m_overlay") * 100.0
    our_fra_3m_direct = column(fra, "fra3m_direct_hyman") * 100.0
    our_fra_3m_linear = column(fra, "fra3m") * 100.0
    our_on_t = column(ois, "t")
    our_on = column(ois, "fwd1d") * 100.0
    our_on_overlay = column(ois, "fwd1d_overlay") * 100.0

    lines = ["Digitized comparison against the published figures (200 dpi page renders, blue curves",
             "extracted by pixel classification, plot frames calibrated on gridlines; digitized",
             "residuals are replica minus source in bp. year-end windows cover the turn-spanning",
             "accruals only. Variants: plain replica, the year-end overlay columns and the direct",
             "turn-knot columns (explicit knots at 27 Dec 2013 / 5 Jan 2014 with a measured",
             "turn-spanning quote, no overlay).",
             "",
             "Short-end instrument note: the 1M/3M/6M/12M curves are bootstrapped from the",
             "source's selected short-end instruments (deposits, the tomorrow FRA and the eight",
             "convexity-corrected 3M futures for 3M, and the 6M/12M FRA strips), not the earlier",
             "synthetic-deposit example table, so the remaining short-end residual reflects curve",
             "shape rather than instrument selection.",
             ""]
    lines.append("Year-end windows:")
    for key, windows in TURN_WINDOWS.items():
        entries = ", ".join(f"{label} [{begin:.2f},{end:.2f}]y"
                            for label, begin, end in windows)
        lines.append(f"  {key}: {entries}")
    lines.append("")

    lines.append("Fig 30 (3M FRA) vs quantape Hyman exogenous:")
    report("fig30_top", digits["fig30_top"], our_fra_t, our_fra_3m, 2.0,
           TURN_WINDOWS["fig30"], "plain replica 0-2Y", lines, "plain", "fig30")
    report("fig30_top", digits["fig30_top"], our_fra_t, our_fra_3m_overlay, 2.0,
           TURN_WINDOWS["fig30"], "year-end overlay 0-2Y", lines, "overlay", "fig30")
    report("fig30_top", digits["fig30_top"], our_fra_t, our_fra_3m_direct, 2.0,
           TURN_WINDOWS["fig30"], "direct turn knots 0-2Y", lines, "direct", "fig30")
    report("fig30_bottom", digits["fig30_bottom"], our_fra_t, our_fra_3m, 50.0,
           TURN_WINDOWS["fig30"], "plain replica 0-50Y", lines, "plain")
    report("fig30_bottom", digits["fig30_bottom"], our_fra_t, our_fra_3m_overlay, 50.0,
           TURN_WINDOWS["fig30"], "year-end overlay 0-50Y", lines, "overlay")
    report("fig30_bottom", digits["fig30_bottom"], our_fra_t, our_fra_3m_direct, 50.0,
           TURN_WINDOWS["fig30"], "direct turn knots 0-50Y", lines, "direct")
    report("fig30_bottom", digits["fig30_bottom"], our_fra_t, our_fra_3m_linear, 50.0,
           TURN_WINDOWS["fig30"], "linear replica 0-50Y", lines, "linear")

    lines.append("Fig 26 (ON FRA) vs quantape OIS:")
    report("fig26_top", digits["fig26_top"], our_on_t, our_on, 2.0,
           TURN_WINDOWS["fig26"], "plain replica 0-2Y", lines, "plain")
    report("fig26_top", digits["fig26_top"], our_on_t, our_on_overlay, 2.0,
           TURN_WINDOWS["fig26"], "year-end overlay 0-2Y", lines, "overlay")
    report("fig26_bottom", digits["fig26_bottom"], our_on_t, our_on, 30.0,
           TURN_WINDOWS["fig26"], "plain replica 0-30Y", lines, "plain")
    report("fig26_bottom", digits["fig26_bottom"], our_on_t, our_on_overlay, 30.0,
           TURN_WINDOWS["fig26"], "year-end overlay 0-30Y", lines, "overlay")

    lines.append("Fig 28 (1M FRA) vs quantape 1M forecast:")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m, 1.0,
           TURN_WINDOWS["fig28"], "plain replica 0-1Y", lines, "plain")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m_overlay, 1.0,
           TURN_WINDOWS["fig28"], "year-end overlay 0-1Y", lines, "overlay")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m_direct, 1.0,
           TURN_WINDOWS["fig28"], "direct turn knots 0-1Y", lines, "direct")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m, 2.0,
           TURN_WINDOWS["fig28"], "plain replica 0-2Y", lines, "plain", "fig28")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m_overlay, 2.0,
           TURN_WINDOWS["fig28"], "year-end overlay 0-2Y", lines, "overlay", "fig28")
    report("fig28_top", digits["fig28_top"], our_fra_t, our_fra_1m_direct, 2.0,
           TURN_WINDOWS["fig28"], "direct turn knots 0-2Y", lines, "direct", "fig28")

    lines.append("")
    lines.append("End-2013 turn-window residual summary (replica minus source, bp):")
    for key in ("fig28", "fig30"):
        for variant in ("plain", "overlay", "direct"):
            entries = [entry for entry in turn_summary if entry[0] == key and entry[1] == variant]
            for _, _, mean, maximum in entries:
                lines.append(f"  {key} {variant:7s} mean|d|={mean:.2f} bp max|d|={maximum:.2f} bp")

    print("\n".join(lines))
    with open(os.path.join(output, "comparison.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0))
    for idx, name in enumerate(("fig30_top", "fig30_bottom")):
        samples = np.array(digits[name])
        axes[idx].plot(samples[:, 0], samples[:, 1], ".", markersize=1.5,
                       label="source (digitized)")
        axes[idx].plot(our_fra_t, our_fra_3m, label="quantape Hyman")
        axes[idx].plot(our_fra_t, our_fra_3m_overlay, label="quantape Hyman + overlay")
        axes[idx].plot(our_fra_t, our_fra_3m_direct, label="quantape Hyman + direct turn knots")
        axes[idx].set_xlabel("years")
        axes[idx].set_ylabel("3M FRA (%)")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    axes[0].set_xlim(0.0, 2.0)
    axes[1].set_xlim(0.0, 50.0)
    axes[0].set_title("EUR 3M FRA: source figure 30 vs replica")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "overlay_3m.png"), dpi=130)

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0))
    for idx, name in enumerate(("fig26_top", "fig26_bottom")):
        samples = np.array(digits[name])
        axes[idx].plot(samples[:, 0], samples[:, 1], ".", markersize=1.5,
                       label="source (digitized)")
        axes[idx].plot(our_on_t, our_on, label="quantape OIS 1D forward")
        axes[idx].plot(our_on_t, our_on_overlay, label="quantape OIS + overlay")
        axes[idx].set_xlabel("years")
        axes[idx].set_ylabel("ON FRA (%)")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    axes[0].set_xlim(0.0, 2.0)
    axes[1].set_xlim(0.0, 30.0)
    axes[0].set_title("EUR ON FRA: source figure 26 vs replica")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "overlay_on.png"), dpi=130)

    samples = np.array(digits["fig28_top"])
    figure, axes = plt.subplots(1, 1, figsize=(10.0, 4.5))
    axes.plot(samples[:, 0], samples[:, 1], ".", markersize=1.5, label="source (digitized)")
    axes.plot(our_fra_t, our_fra_1m, label="quantape 1M")
    axes.plot(our_fra_t, our_fra_1m_overlay, label="quantape 1M + overlay")
    axes.plot(our_fra_t, our_fra_1m_direct, label="quantape 1M + direct turn knots")
    axes.set_xlim(0.0, 2.0)
    axes.set_xlabel("years")
    axes.set_ylabel("1M FRA (%)")
    axes.grid(True, alpha=0.3)
    axes.legend()
    axes.set_title("EUR 1M FRA: source figure 28 vs replica")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "overlay_1m.png"), dpi=130)

    zoom_series = (
        ("fig26_top", our_on_t, our_on, our_on_overlay, None, "ON", TURN_WINDOWS["fig26"]),
        ("fig28_top", our_fra_t, our_fra_1m, our_fra_1m_overlay, our_fra_1m_direct, "1M",
         TURN_WINDOWS["fig28"]),
        ("fig30_top", our_fra_t, our_fra_3m, our_fra_3m_overlay, our_fra_3m_direct, "3M",
         TURN_WINDOWS["fig30"]),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 10.0))
    residual_rows = []
    for ax, (panel, our_t, base_our, overlay_our, direct_our, tenor, windows) in zip(axes,
                                                                                    zoom_series):
        samples = np.array(digits[panel])
        keep = (samples[:, 0] >= 0.75) & (samples[:, 0] <= 1.12)
        window = (our_t >= 0.75) & (our_t <= 1.12)
        ax.plot(samples[keep, 0], samples[keep, 1], "k.", markersize=2.0,
                label="source (digitized)")
        ax.plot(our_t[window], base_our[window], label="replica")
        ax.plot(our_t[window], overlay_our[window], label="replica + overlay")
        if direct_our is not None:
            ax.plot(our_t[window], direct_our[window], label="replica + direct turn knots")
        for _, begin, end in windows:
            ax.axvspan(begin, end, color="tab:red", alpha=0.06)
        ax.set_xlim(0.75, 1.12)
        ax.set_ylabel(f"{tenor} FRA (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        paper_x, paper_y, base_vals, base_diff = residual(digits[panel], our_t, base_our, 2.0)
        _, _, overlay_vals, overlay_diff = residual(digits[panel], our_t, overlay_our, 2.0)
        if direct_our is None:
            direct_vals = np.full_like(base_vals, np.nan)
            direct_diff = np.full_like(base_diff, np.nan)
        else:
            _, _, direct_vals, direct_diff = residual(digits[panel], our_t, direct_our, 2.0)
        inside = np.zeros(paper_x.shape, dtype=bool)
        for _, begin, end in windows:
            inside |= window_mask(paper_x, begin, end)
        for values in zip(paper_x[inside], paper_y[inside], base_vals[inside],
                          overlay_vals[inside], direct_vals[inside], base_diff[inside],
                          overlay_diff[inside], direct_diff[inside]):
            residual_rows.append((panel,) + values)
    axes[0].set_title("Turn windows: source figures vs replica, overlay and direct turn knots")
    axes[-1].set_xlabel("years")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "overlay_turn_zoom.png"), dpi=130)

    with open(os.path.join(output, "year_end_residuals.csv"), "w", newline="") as handle:
        handle.write("panel,t,source_percent,replica_percent,overlay_percent,direct_percent,"
                     "replica_minus_source_bp,overlay_minus_source_bp,direct_minus_source_bp\n")
        for (panel, t, source, replica, overlay, direct, base_diff, overlay_diff,
             direct_diff) in residual_rows:
            handle.write(f"{panel},{t:.6f},{source:.6f},{replica:.6f},{overlay:.6f},"
                         f"{direct:.6f},{base_diff:.4f},{overlay_diff:.4f},{direct_diff:.4f}\n")

    # Diagnostics panel: the turn zooms for the 1M and 3M strips next to the
    # end-2013 residual magnitudes of the three variants.
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    for ax, (panel, our_t, direct_our, tenor, windows) in zip(
            (axes[0], axes[1]),
            (("fig28_top", our_fra_t, our_fra_1m_direct, "1M", TURN_WINDOWS["fig28"]),
             ("fig30_top", our_fra_t, our_fra_3m_direct, "3M", TURN_WINDOWS["fig30"]))):
        samples = np.array(digits[panel])
        keep = (samples[:, 0] >= 0.90) & (samples[:, 0] <= 1.10)
        window = (our_t >= 0.90) & (our_t <= 1.10)
        ax.plot(samples[keep, 0], samples[keep, 1], "k.", markersize=2.0,
                label="source (digitized)")
        base = our_fra_1m if tenor == "1M" else our_fra_3m
        overlay = our_fra_1m_overlay if tenor == "1M" else our_fra_3m_overlay
        ax.plot(our_t[window], base[window], label="plain")
        ax.plot(our_t[window], overlay[window], label="overlay")
        ax.plot(our_t[window], direct_our[window], label="direct turn knots")
        for _, begin, end in windows:
            ax.axvspan(begin, end, color="tab:red", alpha=0.06)
        ax.set_xlim(0.90, 1.10)
        ax.set_ylabel(f"{tenor} FRA (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    bar_ax = axes[2]
    width = 0.35
    positions = np.arange(2)
    for offset, variant, color in ((-width / 2, "plain", "tab:blue"),
                                   (0.0, "overlay", "tab:orange"),
                                   (width / 2, "direct", "tab:green")):
        means = [next(entry[2] for entry in turn_summary
                      if entry[0] == key and entry[1] == variant) for key in ("fig28", "fig30")]
        maxima = [next(entry[3] for entry in turn_summary
                       if entry[0] == key and entry[1] == variant) for key in ("fig28", "fig30")]
        bar_ax.bar(positions + offset, means, width, label=f"{variant} mean", color=color,
                   alpha=0.9)
        bar_ax.plot(positions + offset, maxima, "k_", markersize=9, label=f"{variant} max")
    bar_ax.set_xticks(positions, ["1M", "3M"])
    bar_ax.set_ylabel("end-2013 residual (bp)")
    bar_ax.set_title("Turn-window residual magnitudes")
    bar_ax.grid(True, alpha=0.3)
    bar_ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(os.path.join(output, "turn_diagnostics.png"), dpi=130)

    print("wrote overlay_1m.png, overlay_3m.png, overlay_on.png, overlay_turn_zoom.png, "
          "turn_diagnostics.png, year_end_residuals.csv to", output)


if __name__ == "__main__":
    main()
