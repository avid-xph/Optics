"""Aggregate the Nyquist (task 6) GPU scan into PSNR-vs-cell tables.

Reads results/ablation/nyquist_s{S}_d{D}.csv and prints, for every (sigma, shift):
  * 6-scene mean PSNR per (method, cell)
  * the best cell per method
  * T-free minus ADMM-TV gap per cell

Averaging matches the report convention: mean over windows per scene, then mean
over the 6 scenes. Cells/scenes with no rows are skipped.

Usage:
  python report/analyze_nyquist.py
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ablation")

SIGMAS = [0, 0.5, 1.0]
SHIFTS = [0, 0.25, 0.5]
CELLS = [1, 2, 3, 4, 6, 8]
METHODS = ["admm_tv", "tfree"]
LABEL = {"admm_tv": "ADMM-TV", "tfree": "T-free"}


def load(sigma, shift):
    """-> {method: {cell: 6-scene mean PSNR}}"""
    fn = os.path.join(RES, f"nyquist_s{sigma}_d{shift}.csv")
    if not os.path.exists(fn):
        return {}
    per = defaultdict(lambda: defaultdict(list))   # method -> cell -> scene -> psnr
    with open(fn, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["err"] or r["method"] not in METHODS:
                continue
            if r["cell"] != str(int(r["cell"])):
                continue
            per[r["method"]][int(r["cell"])].append(
                (r["scene"], float(r["psnr"])))
    out = {}
    for m, cells in per.items():
        d = {}
        for c, rows in cells.items():
            by_scene = defaultdict(list)
            for sc, p in rows:
                by_scene[sc].append(p)
            if len(by_scene) == 6:
                d[c] = sum(sum(v) / len(v) for v in by_scene.values()) / 6
        out[m] = d
    return out


def main():
    print("=" * 78)
    print("Nyquist scan (T=8, 6-scene mean PSNR dB)")
    print("=" * 78)
    for sigma in SIGMAS:
        for shift in SHIFTS:
            data = load(sigma, shift)
            if not data:
                print(f"\n[sigma={sigma} shift={shift}]  (missing / not run yet)")
                continue
            print(f"\n[sigma={sigma}  shift={shift}]")
            hdr = f"{'cell':>5} " + "".join(f"{LABEL[m]:>12}" for m in METHODS)
            hdr += f"{'T-free - ADMM':>16}"
            print(hdr)
            print("-" * len(hdr))
            best = {m: (None, -1e9) for m in METHODS}
            for c in CELLS:
                vals = {m: data.get(m, {}).get(c) for m in METHODS}
                if all(v is None for v in vals.values()):
                    continue
                cells_txt = f"{c:>5} "
                for m in METHODS:
                    v = vals[m]
                    cells_txt += f"{v:>12.2f}" if v is not None else f"{'-':>12}"
                a, t = vals["admm_tv"], vals["tfree"]
                if a is not None and t is not None:
                    cells_txt += f"{t - a:>+16.2f}"
                else:
                    cells_txt += f"{'-':>16}"
                print(cells_txt)
                for m in METHODS:
                    v = vals[m]
                    if v is not None and v > best[m][1]:
                        best[m] = (c, v)
            print("-" * len(hdr))
            for m in METHODS:
                c, v = best[m]
                if c is not None:
                    print(f"  best {LABEL[m]:<9}: cell={c}  {v:.2f} dB")


if __name__ == "__main__":
    main()
