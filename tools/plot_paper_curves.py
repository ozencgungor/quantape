#!/usr/bin/env python3
"""Plot the EUR benchmark curve replication outputs.

Reads the CSV files produced by

    QTA_PAPER_EUR_DUMP=<directory> ./build/release/tests/test_paper_eur_curves

and writes comparison figures into the same directory.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows, name):
    return [float(row[name]) for row in rows]


def bp(x):
    return [value * 1e4 for value in x]


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "internal_docs/paper_fit"
    ois = read_rows(os.path.join(directory, "ois.csv"))
    fra = read_rows(os.path.join(directory, "fra.csv"))
    exo = read_rows(os.path.join(directory, "exo_endog.csv"))
    interp = read_rows(os.path.join(directory, "interpolation.csv"))
    imm = read_rows(os.path.join(directory, "imm_one_year_out_of_sample.csv"))
    pillars = read_rows(os.path.join(directory, "pillars_3m.csv"))

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0))
    short = [row for row in ois if float(row["t"]) <= 2.0]
    axes[0].plot(values(short, "t"), values(short, "zero"), label="zero rate")
    axes[0].plot(values(short, "t"), values(short, "fwd1d"), label="1D forward")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("EUR OIS spot curve, short end (11 Dec 2012)")
    axes[0].set_xlabel("years")
    axes[0].set_ylabel("rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(values(ois, "t"), values(ois, "zero"), label="zero rate")
    axes[1].plot(values(ois, "t"), values(ois, "fwd3m"), label="3M forward")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("EUR OIS spot curve, full term structure")
    axes[1].set_xlabel("years")
    axes[1].set_ylabel("rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(directory, "ois.png"), dpi=130)

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0))
    short = [row for row in fra if float(row["t"]) <= 2.0]
    axes[0].plot(values(short, "t"), bp(values(short, "fra3m")), label="3M FRA")
    axes[0].plot(values(short, "t"), bp(values(short, "fra1m")), label="1M FRA")
    axes[0].plot(values(short, "t"), bp(values(short, "ois3m")), label="OIS 3M forward")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("EUR FRA curves, short end (bp)")
    axes[0].set_xlabel("years")
    axes[0].set_ylabel("bp")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(values(fra, "t"), bp(values(fra, "fra3m")), label="3M FRA")
    axes[1].plot(values(fra, "t"), bp(values(fra, "fra1m")), label="1M FRA")
    axes[1].plot(values(fra, "t"), bp(values(fra, "ois3m")), label="OIS 3M forward")
    axes[1].plot(values(pillars, "t"), values(pillars, "quote_bp"), "o", markersize=3.0,
                 label="3M IRS pillars")
    axes[1].set_title("EUR FRA curves and OIS, full term structure (bp)")
    axes[1].set_xlabel("years")
    axes[1].set_ylabel("bp")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(directory, "fra.png"), dpi=130)

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0][0].plot(values(fra, "t"), bp(values(fra, "basis3m")))
    axes[0][0].axhline(0.0, color="black", linewidth=0.8)
    axes[0][0].set_title("3M FRA minus OIS 3M forward (bp)")
    axes[0][0].set_xlabel("years")
    axes[0][0].grid(True, alpha=0.3)
    axes[0][1].plot(values(exo, "t"), values(exo, "difference_bp"))
    axes[0][1].axhline(0.0, color="black", linewidth=0.8)
    axes[0][1].set_title("Exogenous minus endogenous 3M FRA (bp)")
    axes[0][1].set_xlabel("years")
    axes[0][1].grid(True, alpha=0.3)
    axes[1][0].plot(values(interp, "t"), bp(values(interp, "linear")), label="linear")
    axes[1][0].plot(values(interp, "t"), bp(values(interp, "hyman")), label="Hyman spline")
    axes[1][0].plot(values(interp, "t"), bp(values(interp, "monotone")), label="monotone cubic")
    axes[1][0].set_title("Endogenous 3M FRA, interpolation schemes (bp)")
    axes[1][0].set_xlabel("years")
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].legend()
    labels = [row["start"] for row in imm]
    residuals = values(imm, "difference_bp")
    axes[1][1].bar(range(len(residuals)), residuals)
    axes[1][1].axhline(0.0, color="black", linewidth=0.8)
    axes[1][1].set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    axes[1][1].set_title("IMM forward-start 1Y swaps: model minus quote (bp)")
    axes[1][1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(directory, "diagnostics.png"), dpi=130)

    print("wrote ois.png, fra.png, diagnostics.png to", directory)


if __name__ == "__main__":
    main()
