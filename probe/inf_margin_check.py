"""Compare the historical evaluation threshold with a strict overload threshold.

Report aggregate sensitivity to numerical tolerance for one snapshot.
Usage: python inf_margin_check.py SNAPSHOT OUTPUT"""
import json
import os
import sys

import numpy as np

import cac_core as cc

CAP, BOT = cc.CAP, cc.BOT_RATE
LOADS, CAPS, RHOS, DELTAS = (0.05, 0.1), (20, 50), (1.0, 0.9), (1, 2)


def strict_cost(ps, pl, x, R, B, l):
    """Greedy cost (Mbps) to push GSL l strictly above its threshold, no tolerance; inf if impossible."""
    sel = np.where((pl == l) & (x > 1e-12) & (B[ps] > 0))[0]
    sel = sel[np.argsort(-x[sel], kind='stable')]
    acc = cost = 0.0
    for i in sel:
        gain = B[ps[i]] * x[i]
        if acc + gain > R:
            return cost + (R - acc) / x[i]
        acc += gain
        cost += B[ps[i]]
    return np.inf


def examine(ps, pl, x, d, B, rho, landing, nl):
    legit = cc.legit_load(ps, pl, x, d, nl)
    cprime = CAP / rho
    R = cprime - legit[landing]
    use = (x > 1e-12) & (B[ps] > 0)
    A = np.bincount(pl[use], weights=B[ps[use]] * x[use], minlength=nl)[landing]
    cong_cur = (R <= 0) | (A > R * (1 + 1e-7) + 1e-6)
    cong_str = (R < 0) | (A > R)
    cost_cur = cc.certified_cost(ps, pl, x, legit, B, cprime, landing)
    flip = cong_str & ~cong_cur
    cost_str = cost_cur.copy()
    for i in np.where(flip)[0]:
        cost_str[i] = strict_cost(ps, pl, x, R[i], B, landing[i])
    slack = R - A
    rec = {'cert_cur': (float(cost_cur.min()) / BOT) if np.isfinite(cost_cur.min()) else None,
           'cert_str': (float(cost_str.min()) / BOT) if np.isfinite(cost_str.min()) else None,
           'unc_cur': bool(not cong_cur.any()), 'unc_str': bool(not cong_str.any()),
           'n_flip_gsl': int(flip.sum()),
           'max_overshoot_mbps': float((A - R)[flip].max()) if flip.any() else 0.0,
           'n_R_zero': int((R == 0).sum()), 'n_R_tiny': int((np.abs(R) <= 1e-6).sum()),
           'n_R_le0_A0': int(((R <= 0) & (A == 0)).sum())}
    if rec['unc_cur']:
        rec['min_slack_mbps'] = float(slack.min())
    return rec


def main():
    snapfile, out = sys.argv[1], sys.argv[2]
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

    pl_sp = land_of[src]
    ps0, pl0, _ = cc.build_candidates(src, landing, H, 0)
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
    fixed = {'SP': (src, pl_sp, np.ones(len(src))), 'ECMP': (ps0, pl0, x_ecmp), 'PCDR': (ps_p, pl_p, x_p)}

    rows = []
    for load in LOADS:
        d = d0 * load
        mlu = {}
        for delta in DELTAS:
            psd, pld, phd = cc.build_candidates(src, landing, H, delta)
            xm, _ = cc.solve_mlu(src, psd, pld, phd, d, landing)
            mlu[delta] = (psd, pld, phd, xm)
        for cap in CAPS:
            B = np.where(has_pop, np.minimum(cap * BOT, np.maximum(0.0, CAP - d)), 0.0)
            for rho in RHOS:
                for delta in DELTAS:
                    psd, pld, phd, xm = mlu[delta]
                    xc, _, unc = cc.cac_bisect(src, psd, pld, phd, d, B, landing, CAP / rho)
                    routs = dict(fixed)
                    routs['MLU'] = (psd, pld, xm)
                    if xc is not None:
                        routs['CAC'] = (psd, pld, xc)
                    for name, (ps, pl, x) in routs.items():
                        rec = examine(ps, pl, x, d, B, rho, landing, nl)
                        rec.update({'load': load, 'cap': cap, 'rho': rho, 'delta': delta, 'routing': name})
                        if name == 'CAC':
                            rec['lp_uncongestible'] = bool(unc)
                        rows.append(rec)
    json.dump({'snap': os.path.basename(snapfile), 'rows': rows}, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
