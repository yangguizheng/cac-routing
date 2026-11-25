"""Evaluate routing policies for one fixed topology and expected source demand.

Usage: python ts_eval.py SNAPSHOT OUTPUT LOADS CAPS RHOS DELTAS
The last four arguments are comma-separated lists."""
import json
import os
import sys
import time

import numpy as np

import cac_core as cc

CAP, BOT = cc.CAP, cc.BOT_RATE


def main():
    snapfile, out = sys.argv[1], sys.argv[2]
    loads = [float(v) for v in sys.argv[3].split(',')]
    caps = [int(v) for v in sys.argv[4].split(',')]
    rhos = [float(v) for v in sys.argv[5].split(',')]
    deltas = [int(v) for v in sys.argv[6].split(',')]
    snap = np.load(snapfile)
    n_orb, n_per = int(snap['n_orb']), int(snap['n_per'])
    n_sat = n_orb * n_per
    ucs, scg, land_of, pop = snap['ucs'], snap['scg'], snap['land_of'], snap['pop']
    w = np.where(pop != 0, np.ceil(pop), 0.0)
    off = 1e6 * w / w.sum()
    d0 = np.bincount(ucs[ucs >= 0], weights=off[ucs >= 0], minlength=n_sat)
    has_pop = np.zeros(n_sat, bool)
    has_pop[np.unique(ucs[(ucs >= 0) & (pop > 0)])] = True
    landing = np.where(scg >= 0)[0]
    src = np.where((d0 > 0) | has_pop)[0]
    H = cc.hops_matrix(src, landing, n_orb, n_per)
    nl = int(landing.max()) + 1
    rows = []

    def cert(ps, pl, x, d, B, rho):
        legit = cc.legit_load(ps, pl, x, d, nl)
        c = cc.certified_cost(ps, pl, x, legit, B, CAP / rho, landing)
        return (float(c.min()) / BOT) if np.isfinite(c.min()) else None, float(legit[landing].max() / CAP)

    # fixed routings
    pl_sp = land_of[src]
    ph_sp = H[np.arange(len(src)), np.searchsorted(landing, pl_sp)].astype(float)
    ps0, pl0, ph0 = cc.build_candidates(src, landing, H, 0)
    x_ecmp = 1.0 / np.bincount(ps0, minlength=n_sat)[ps0]
    acc = {}
    for s_ in src:
        l_ = land_of[s_]
        o, k = l_ // n_per, l_ % n_per
        cand = [l_, o * n_per + (k + 1) % n_per, o * n_per + (k - 1) % n_per,
                ((o + 1) % n_orb) * n_per + k, ((o - 1) % n_orb) * n_per + k]
        cand = [c for c in cand if scg[c] == scg[l_]]
        for c in cand:
            acc[(s_, c)] = acc.get((s_, c), 0.0) + 1.0 / len(cand)
    keys = list(acc.keys())
    ps_p = np.array([k_[0] for k_ in keys])
    pl_p = np.array([k_[1] for k_ in keys])
    x_p = np.array([acc[k_] for k_ in keys])
    ph_p = H[np.searchsorted(src, ps_p), np.searchsorted(landing, pl_p)].astype(float)

    for load in loads:
        d = d0 * load
        mlu = {}
        for delta in deltas:
            psd, pld, phd = cc.build_candidates(src, landing, H, delta)
            t0 = time.time()
            xm, u = cc.solve_mlu(src, psd, pld, phd, d, landing)
            mlu[delta] = (psd, pld, phd, xm, u, time.time() - t0)
        for cap in caps:
            B = np.where(has_pop, np.minimum(cap * BOT, np.maximum(0.0, CAP - d)), 0.0)
            for rho in rhos:
                base = {}
                for name, (ps, pl, x, ph) in {'SP': (src, pl_sp, np.ones(len(src)), ph_sp),
                                              'ECMP': (ps0, pl0, x_ecmp, ph0),
                                              'PCDR': (ps_p, pl_p, x_p, ph_p)}.items():
                    c, u = cert(ps, pl, x, d, B, rho)
                    base[name] = {'cert_bots': c, 'max_util': u,
                                  'mean_hops': float((d[ps] * x * ph).sum() / d.sum())}
                for delta in deltas:
                    psd, pld, phd, xm, um, tm = mlu[delta]
                    cm, _ = cert(psd, pld, xm, d, B, rho)
                    t0 = time.time()
                    xc, T, unc = cc.cac_bisect(src, psd, pld, phd, d, B, landing, CAP / rho)
                    tc = time.time() - t0
                    row = {'load': load, 'cap': cap, 'rho': rho, 'delta': delta, **{f'{k}_{kk}': vv for k, v in base.items() for kk, vv in v.items()},
                           'MLU_cert_bots': cm, 'MLU_max_util': um, 'MLU_seconds': tm,
                           'MLU_mean_hops': float((d[psd] * xm * phd).sum() / d.sum())}
                    if xc is None:
                        row.update({'CAC_status': 'legal load does not fit'})
                    else:
                        cc_, uc = cert(psd, pld, xc, d, B, rho)
                        row.update({'CAC_status': 'ok', 'CAC_T_bots': T / BOT, 'CAC_uncongestible': unc,
                                    'CAC_cert_bots': cc_, 'CAC_max_util': uc, 'CAC_seconds': tc,
                                    'CAC_mean_hops': float((d[psd] * xc * phd).sum() / d.sum())})
                        if rho == rhos[0] and cap == caps[-1]:
                            row['CAC_isl_max_util'] = cc.isl_max_util(psd, pld, xc, d, n_orb, n_per)
                            row['MLU_isl_max_util'] = cc.isl_max_util(psd, pld, xm, d, n_orb, n_per)
                    rows.append(row)
    json.dump({'snap': os.path.basename(snapfile), 'n_src': int(len(src)), 'n_landing': int(len(landing)), 'rows': rows,
               'lp': cc.STATS['lp'], 'lp_timeouts': cc.STATS['timeouts']}, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
