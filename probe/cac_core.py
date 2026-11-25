"""Linear programs and capacity certificates for routing on a periodic orbital grid.

Rates are expressed in Mbps at the interface and in Gbps inside the solver."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

CAP = 4096.0          # GSL capacity, Mbps
CAP_ISL = 20480.0     # ISL capacity, Mbps
BOT_RATE = 20.0       # Mbps per compromised terminal
G = 1000.0            # Mbps -> Gbps inside LPs
LP_OPTS = {'time_limit': 300.0}
STATS = {'lp': 0, 'timeouts': 0}


def hops_matrix(src, dst, n_orb, n_per):
    so, sk = src // n_per, src % n_per
    do, dk = dst // n_per, dst % n_per
    od = np.abs(so[:, None] - do[None, :])
    od = np.minimum(od, n_orb - od)
    kd = np.abs(sk[:, None] - dk[None, :])
    kd = np.minimum(kd, n_per - kd)
    return od + kd


def build_candidates(src, landing, H, delta):
    hmin = H.min(1)
    ps, pl, ph = [], [], []
    for i, s in enumerate(src):
        sel = np.where(H[i] <= hmin[i] + delta)[0]
        ps.extend([s] * len(sel))
        pl.extend(landing[sel].tolist())
        ph.extend(H[i, sel].tolist())
    return np.array(ps), np.array(pl), np.array(ph, dtype=float)


def grid_path_links(s, l, n_orb, n_per):
    """Inter-orbit moves first, then intra-orbit, shorter way round; SKYFALL link ids (6*i+k)."""
    n_sat = n_orb * n_per
    links = []
    cur = s
    so, lo_ = s // n_per, l // n_per
    step = 0
    if so != lo_:
        fwd = (lo_ - so) % n_orb
        direction = 1 if fwd <= n_orb - fwd else -1
        while cur // n_per != lo_:
            nxt = (cur + direction * n_per) % n_sat
            links.append(cur * 6 + (1 if direction == 1 else 3))
            cur = nxt
            step += 1
            assert step < 500
    sk, lk = cur % n_per, l % n_per
    if sk != lk:
        fwd = (lk - sk) % n_per
        direction = 1 if fwd <= n_per - fwd else -1
        while cur != l:
            o, k = cur // n_per, cur % n_per
            nxt = o * n_per + (k + direction) % n_per
            links.append(cur * 6 + (0 if direction == 1 else 2))
            cur = nxt
            step += 1
            assert step < 500
    return links


def isl_max_util(ps, pl, x, d, n_orb, n_per):
    load = {}
    cache = {}
    for s, l, xx in zip(ps, pl, x):
        if xx <= 1e-9 or d[s] <= 0 or s == l:
            continue
        key = (int(s), int(l))
        if key not in cache:
            cache[key] = grid_path_links(int(s), int(l), n_orb, n_per)
        for lk in cache[key]:
            load[lk] = load.get(lk, 0.0) + d[s] * xx
    return (max(load.values()) / CAP_ISL) if load else 0.0


def certified_cost(pairs_s, pairs_l, r, legit_l, B, cprime, landing):
    """Return each downlink capacity certificate in Mbps; infinity denotes no overload.

    The overload test includes an absolute margin of 1e-6 Mbps and a relative
    margin of 1e-7 on nonnegative spare capacity.
    """
    out = np.full(len(landing), np.inf)
    order = np.lexsort((-r, pairs_l))
    pairs_s, pairs_l, r = pairs_s[order], pairs_l[order], r[order]
    lo = np.searchsorted(pairs_l, landing, side='left')
    hi = np.searchsorted(pairs_l, landing, side='right')
    for i, l in enumerate(landing):
        R = cprime - legit_l[l]
        if R < 0:
            out[i] = 0.0
            continue
        acc, cost = 0.0, 0.0
        for s, rr in zip(pairs_s[lo[i]:hi[i]], r[lo[i]:hi[i]]):
            if rr <= 1e-12 or B[s] <= 0:
                continue
            if acc + B[s] * rr > R * (1 + 1e-7) + 1e-6:
                cost += (R - acc) / rr
                out[i] = cost
                break
            acc += B[s] * rr
            cost += B[s]
    return out


def legit_load(ps, pl, x, d, n):
    return np.bincount(pl, weights=d[ps] * x, minlength=n)


def _index(src, landing, ps, pl):
    sidx = {s: i for i, s in enumerate(src)}
    lidx = {l: i for i, l in enumerate(landing)}
    return np.array([sidx[s] for s in ps]), np.array([lidx[l] for l in pl])


def solve_mlu(src, ps, pl, ph, d, landing):
    n, nl = len(ps), len(landing)
    rows_eq, li = _index(src, landing, ps, pl)
    dg = d / G
    Aeq = sp.csr_matrix((np.ones(n), (rows_eq, np.arange(n))), shape=(len(src), n + 1))
    Aub = sp.csr_matrix((np.concatenate([dg[ps], np.full(nl, -CAP / G)]),
                         (np.concatenate([li, np.arange(nl)]), np.concatenate([np.arange(n), np.full(nl, n)]))),
                        shape=(nl, n + 1))
    c = np.zeros(n + 1)
    c[n] = 1.0
    STATS['lp'] += 1
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(nl), A_eq=Aeq, b_eq=np.ones(len(src)),
                bounds=[(0, None)] * (n + 1), method='highs', options=LP_OPTS)
    assert r.status == 0, r.message
    u = r.x[n]
    Aub2 = sp.csr_matrix((dg[ps], (li, np.arange(n))), shape=(nl, n))
    STATS['lp'] += 1
    r2 = linprog(dg[ps] * ph + 1e-6 * ph, A_ub=Aub2, b_ub=np.full(nl, u * CAP / G * 1.0001),
                 A_eq=Aeq[:, :n], b_eq=np.ones(len(src)), bounds=[(0, None)] * n, method='highs', options=LP_OPTS)
    assert r2.status == 0, r2.message
    return r2.x, u


def cac_lp_vec(src, ps, pl, ph, d, B, landing, cprime, Tvec, legit_cap=CAP):
    """Feasibility LP: every GSL l withstands a malicious budget Tvec[l] (Mbps). Returns x or None."""
    n, nl = len(ps), len(landing)
    rows_eq, li = _index(src, landing, ps, pl)
    dg, Bg, Tg = d / G, B / G, np.asarray(Tvec) / G
    nv = 2 * n + nl
    Aeq = sp.csr_matrix((np.ones(n), (rows_eq, np.arange(n))), shape=(len(src), nv))
    A1 = sp.csr_matrix((np.concatenate([dg[ps], Tg, Bg[ps]]),
                        (np.concatenate([li, np.arange(nl), li]),
                         np.concatenate([np.arange(n), n + np.arange(nl), n + nl + np.arange(n)]))), shape=(nl, nv))
    k = np.arange(n)
    A2 = sp.csr_matrix((np.concatenate([np.ones(n), -np.ones(n), -np.ones(n)]),
                        (np.concatenate([k, k, k]), np.concatenate([k, n + li, n + nl + k]))), shape=(n, nv))
    A3 = sp.csr_matrix((dg[ps], (li, np.arange(n))), shape=(nl, nv))
    Aub = sp.vstack([A1, A2, A3]).tocsr()
    bub = np.concatenate([np.full(nl, cprime / G), np.zeros(n), np.full(nl, legit_cap / G)])
    c = np.zeros(nv)
    c[:n] = dg[ps] * ph + 1e-6 * ph
    STATS['lp'] += 1
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.ones(len(src)),
                bounds=[(0, None)] * nv, method='highs', options=LP_OPTS)
    if r.status == 1:
        STATS['timeouts'] += 1
    return r.x[:n] if r.status == 0 else None


def reach_cap(ps, pl, B, landing):
    """Largest malicious traffic that can reach each GSL at all (sum of caps of sources that may land there)."""
    return np.bincount(pl, weights=B[ps], minlength=int(landing.max()) + 1)[landing]


def cac_bisect(src, ps, pl, ph, d, B, landing, cprime, fixed=None, Tfixed=None, lo=0.0, rel=0.01, legit_cap=CAP):
    """Max common budget t for non-fixed GSLs (fixed ones keep Tfixed). Returns (x, t, uncongestible_all)."""
    nl = len(landing)
    fixed = np.zeros(nl, bool) if fixed is None else fixed
    Tfixed = np.zeros(nl) if Tfixed is None else Tfixed
    hi = float(reach_cap(ps, pl, B, landing)[~fixed].max()) * 1.0001 + 1.0 if (~fixed).any() else lo

    def feas(t):
        return cac_lp_vec(src, ps, pl, ph, d, B, landing, cprime, np.where(fixed, Tfixed, t), legit_cap)

    best = feas(lo)
    if best is None:
        return None, None, False
    top = feas(hi)
    if top is not None:
        return top, hi, True
    while hi - lo > max(10.0, rel * lo):
        mid = 0.5 * (lo + hi)
        xx = feas(mid)
        if xx is None:
            hi = mid
        else:
            lo, best = mid, xx
    return best, lo, False


def lex_cac(src, ps, pl, ph, d, B, landing, cprime, levels=3, tol=0.02):
    """Level-by-level max-min: raise the weakest GSLs, fix the binding ones, raise the rest."""
    nl = len(landing)
    fixed = np.zeros(nl, bool)
    Tfixed = np.zeros(nl)
    x, info = None, []
    lo = 0.0
    for lev in range(levels):
        xx, t, all_unc = cac_bisect(src, ps, pl, ph, d, B, landing, cprime, fixed, Tfixed, lo=lo)
        if xx is None:
            break
        x = xx
        info.append({'level': lev + 1, 't_mbps': t, 'uncongestible_rest': bool(all_unc),
                     'n_fixed_before': int(fixed.sum())})
        if all_unc:
            break
        legit = legit_load(ps, pl, x, d, int(landing.max()) + 1)
        cost = certified_cost(ps, pl, x, legit, B, cprime, landing)
        newly = (~fixed) & (cost <= t * (1 + tol) + 1.0)
        if not newly.any():
            break
        Tfixed[newly] = t * (1 - 1e-4)
        fixed |= newly
        lo = t
    return x, info
