"""Measure demand-weighted propagation-delay spread across active routes.

Usage: python reorder_spread.py SNAPSHOT SKYFALL_DIRECTORY OUTPUT"""
import json
import os
import sys

import numpy as np

import cac_core as cc

CAP, BOT = cc.CAP, cc.BOT_RATE
RADIUS = 6371.0
C_KM_PER_MS = 299.792458
BIN_MS = 0.1
NBIN = 601                      # 0 .. 60 ms
THRESH = (1e-6, 1e-3)           # a landing satellite counts as used if its fraction exceeds this
LOADS, CAPS, RHOS, DELTA = (0.05, 0.1), (20, 50), (1.0, 0.9), 2


def car(lat, lng, h):
    lat = np.radians(lat)
    lng = np.radians(lng)
    r = RADIUS + h
    return np.stack([r * np.cos(lat) * np.cos(lng), r * np.cos(lat) * np.sin(lng), r * np.sin(lat)], axis=-1)


def isl_len_km(s, l, pos, n_orb, n_per):
    n_sat = n_orb * n_per
    tot = 0.0
    for lk in cc.grid_path_links(int(s), int(l), n_orb, n_per):
        i, k = divmod(lk, 6)
        o, kk = divmod(i, n_per)
        if k == 1:
            j = (i + n_per) % n_sat
        elif k == 3:
            j = (i - n_per) % n_sat
        elif k == 0:
            j = o * n_per + (kk + 1) % n_per
        else:
            j = o * n_per + (kk - 1) % n_per
        tot += float(np.linalg.norm(pos[i] - pos[j]))
    return tot


def summarize(ps, pl, x, d, delay, gw_of):
    out = {}
    for th in THRESH:
        use = x > th
        hist = np.zeros(NBIN)
        single_land = single_gw = tot = 0.0
        mx = 0.0
        srcs = np.unique(ps[use])
        for s in srcs:
            m = use & (ps == s)
            dl = delay[m]
            sp = float(dl.max() - dl.min())
            w = float(d[s])
            tot += w
            hist[min(NBIN - 1, int(sp / BIN_MS))] += w
            single_land += w if m.sum() == 1 else 0.0
            single_gw += w if len(set(gw_of[pl[m]].tolist())) == 1 else 0.0
            mx = max(mx, sp)
        out[f'{th:g}'] = {'hist': hist.tolist(), 'tot': tot, 'single_land': single_land,
                          'single_gw': single_gw, 'max_ms': mx}
    return out


def main():
    snapfile, skydir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    snap = np.load(snapfile)
    n_orb, n_per = int(snap['n_orb']), int(snap['n_per'])
    n_sat = n_orb * n_per
    ucs, scg, pop = snap['ucs'], snap['scg'], snap['pop']
    lla = snap['sat_lla']
    pos = car(lla[:, 0], lla[:, 1], lla[:, 2])
    gs_info = json.load(open(os.path.join(skydir, 'skyfall', 'GS.json')))
    keys = list(gs_info.keys())
    gs = car(np.array([float(gs_info[k]['lat']) for k in keys]),
             np.array([float(gs_info[k]['lng']) for k in keys]), 0.0)

    w = np.where(pop != 0, np.ceil(pop), 0.0)
    off = 1e6 * w / w.sum()
    d0 = np.bincount(ucs[ucs >= 0], weights=off[ucs >= 0], minlength=n_sat)
    has_pop = np.zeros(n_sat, bool)
    has_pop[np.unique(ucs[(ucs >= 0) & (pop > 0)])] = True
    landing = np.where(scg >= 0)[0]
    src = np.where((d0 > 0) | has_pop)[0]
    H = cc.hops_matrix(src, landing, n_orb, n_per)
    nl = int(landing.max()) + 1
    psd, pld, phd = cc.build_candidates(src, landing, H, DELTA)

    # one-way propagation per candidate pair: ISL path + GSL slant range (uplink is common to a source)
    delay = np.array([(isl_len_km(s, l, pos, n_orb, n_per) + float(np.linalg.norm(pos[l] - gs[scg[l]]))) / C_KM_PER_MS
                      for s, l in zip(psd, pld)])
    hop_ok = all(len(cc.grid_path_links(int(s), int(l), n_orb, n_per)) == int(h) for s, l, h in zip(psd, pld, phd))

    res = {'snap': os.path.basename(snapfile), 'hop_check': hop_ok, 'rows': []}
    for load in LOADS:
        d = d0 * load
        xm, _ = cc.solve_mlu(src, psd, pld, phd, d, landing)
        res['rows'].append({'routing': 'MLU', 'load': load, **summarize(psd, pld, xm, d, delay, scg)})
        for cap in CAPS:
            B = np.where(has_pop, np.minimum(cap * BOT, np.maximum(0.0, CAP - d)), 0.0)
            for rho in RHOS:
                xc, T, unc = cc.cac_bisect(src, psd, pld, phd, d, B, landing, CAP / rho)
                if xc is None:
                    res['rows'].append({'routing': 'CAC', 'load': load, 'cap': cap, 'rho': rho, 'status': 'nofit'})
                    continue
                legit = cc.legit_load(psd, pld, xc, d, nl)
                c = cc.certified_cost(psd, pld, xc, legit, B, CAP / rho, landing)
                cert = (float(c.min()) / BOT) if np.isfinite(c.min()) else None
                res['rows'].append({'routing': 'CAC', 'load': load, 'cap': cap, 'rho': rho, 'status': 'ok',
                                    'cert_bots': cert, **summarize(psd, pld, xc, d, delay, scg)})
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
