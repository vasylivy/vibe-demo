#!/usr/bin/env python3
"""Plot Q1 MMS convergence from a CSV produced by test_mms_convergence.

Expected CSV (header and all three columns required):

    n,l2_error,h1_error
    10,1.0e-2,5.0e-2
    20,2.5e-3,1.2e-2
    40,6.2e-4,3.0e-3
    80,1.5e-4,7.5e-4

`h = 1/n` on a log-log plot; reference lines for `h^2` (L²) and
`h` (H¹ seminorm) are pinned to each curve's coarsest rung.
Empirical slopes are fit per curve and shown in the legend.

Usage:
    python3 scripts/plot_mms_convergence.py <input.csv> [output.png]

If the output path is omitted, writes alongside the CSV as
mms_convergence.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns, l2, h1 = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"n", "l2_error", "h1_error"}
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {sorted(missing)}; "
                f"have {sorted(fieldnames)}")
        for row in reader:
            ns.append(int(row["n"]))
            l2.append(float(row["l2_error"]))
            h1.append(float(row["h1_error"]))
    return np.asarray(ns), np.asarray(l2), np.asarray(h1)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) > 2 else in_path.with_name("mms_convergence.png")

    ns, l2, h1 = read_csv(in_path)
    if ns.size < 2:
        print(f"need at least 2 rungs, got {ns.size}", file=sys.stderr)
        return 1

    h = 1.0 / ns

    l2_slope, _ = np.polyfit(np.log(h), np.log(l2), 1)
    l2_ref = l2[0] * (h / h[0]) ** 2

    h1_slope, _ = np.polyfit(np.log(h), np.log(h1), 1)
    h1_ref = h1[0] * (h / h[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.loglog(h, l2, "o-", color="C0",
              label=rf"$\|u_h - u\|_{{L^2}}$ (slope ≈ {l2_slope:.2f})")
    ax.loglog(h, l2_ref, "--", color="C0", alpha=0.5,
              label=r"$O(h^{2})$")
    ax.loglog(h, h1, "s-", color="C3",
              label=rf"$|u_h - u|_{{H^1}}$ (slope ≈ {h1_slope:.2f})")
    ax.loglog(h, h1_ref, "--", color="C3", alpha=0.5,
              label=r"$O(h)$")

    ax.set_xlabel("h = 1/N")
    ax.set_ylabel("error")
    ax.set_title(f"Q1 Poisson MMS convergence ({ns.size} rungs)")
    ax.set_xticks(h)
    ax.set_xticklabels([f"1/{n}" for n in ns])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    print(f"wrote {out_path} (L2 {l2_slope:.3f}, H1 {h1_slope:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
