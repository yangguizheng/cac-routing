"""Summarize linear-program dimensions across topology snapshots."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import cac_core as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "step5")
    args = parser.parse_args()
    files = sorted(args.results.glob('ts/snap_*.npz'))
    if not files:
        parser.error('No ts/snap_*.npz files found under the results directory.')
    rows = []
    for f in files:
        s = np.load(f)
        n_orb, n_per = int(s['n_orb']), int(s['n_per'])
        n_sat = n_orb * n_per
        ucs, scg, pop = s['ucs'], s['scg'], s['pop']
        w = np.where(pop != 0, np.ceil(pop), 0.0)
        off = 1e6 * w / w.sum()
        d0 = np.bincount(ucs[ucs >= 0], weights=off[ucs >= 0], minlength=n_sat)
        has_pop = np.zeros(n_sat, bool); has_pop[np.unique(ucs[(ucs >= 0) & (pop > 0)])] = True
        landing = np.where(scg >= 0)[0]
        src = np.where((d0 > 0) | has_pop)[0]
        H = cc.hops_matrix(src, landing, n_orb, n_per)
        r = {'ts': int(os.path.basename(f)[5:-4]), 'n_src': int(len(src)), 'n_land': int(len(landing)),
             'n_gs_used': int(len(np.unique(scg[landing]))), 'hmin_mean': float(H.min(1).mean()),
             'share_bentpipe': float((H.min(1) == 0).mean()), 'demand_covered': float(d0.sum() / 1e6)}
        for dl in (0, 1, 2):
            ps, pl, ph = cc.build_candidates(src, landing, H, dl)
            r[f'P{dl}'] = int(len(ps))
        rows.append(r)
    with (args.results / "lp_size_stats.json").open("w") as output:
        json.dump(rows, output)
    keys = [k for k in rows[0] if k != 'ts']
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        print(f'{k:16s} min {v.min():9.3f} median {np.median(v):9.3f} max {v.max():9.3f}')


if __name__ == "__main__":
    main()
